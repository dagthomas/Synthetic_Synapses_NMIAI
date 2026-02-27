const std = @import("std");
const builtin = @import("builtin");
const net = std.net;
const tls = std.crypto.tls;
const Allocator = std.mem.Allocator;
const is_windows = builtin.os.tag == .windows;

fn setTcpNoDelay(tcp: net.Stream) !void {
    if (is_windows) {
        // Windows: use ws2_32 setsockopt directly
        const ws2 = std.os.windows.ws2_32;
        const IPPROTO_TCP: i32 = 6;
        const TCP_NODELAY: i32 = 1;
        var optval: u32 = 1;
        const result = ws2.setsockopt(tcp.handle, IPPROTO_TCP, TCP_NODELAY, @as([*]const u8, @ptrCast(&optval)), @sizeOf(u32));
        if (result == ws2.SOCKET_ERROR) return error.SetSockOptFailed;
    } else {
        const IPPROTO_TCP: u32 = 6;
        const TCP_NODELAY: u32 = 1;
        std.posix.setsockopt(tcp.handle, IPPROTO_TCP, TCP_NODELAY, &std.mem.toBytes(@as(c_int, 1))) catch return error.SetSockOptFailed;
    }
    std.debug.print("TCP_NODELAY set successfully\n", .{});
}

pub const WsClient = struct {
    stream: net.Stream,
    tls_client: ?*TlsState,
    recv_buf: []u8,
    recv_pos: usize,
    recv_len: usize,
    allocator: Allocator,

    const TlsState = struct {
        client: tls.Client,
        stream_writer: net.Stream.Writer,
        stream_reader: net.Stream.Reader,
        arena: std.heap.ArenaAllocator,
    };

    pub fn connect(allocator: Allocator, host: []const u8, port: u16, path: []const u8, use_tls: bool) !WsClient {
        const tcp = try net.tcpConnectToHost(allocator, host, port);
        errdefer tcp.close();

        // Set TCP_NODELAY to minimize latency (prevent Nagle's algorithm from buffering small frames)
        setTcpNoDelay(tcp) catch |err| {
            std.debug.print("Warning: TCP_NODELAY failed: {any}\n", .{err});
        };

        var self = WsClient{
            .stream = tcp,
            .tls_client = null,
            .recv_buf = try allocator.alloc(u8, 1024 * 1024),
            .recv_pos = 0,
            .recv_len = 0,
            .allocator = allocator,
        };
        errdefer allocator.free(self.recv_buf);

        if (use_tls) {
            // Heap-allocate TLS state so it doesn't move when WsClient is returned
            const tc = try allocator.create(TlsState);
            errdefer allocator.destroy(tc);

            tc.arena = std.heap.ArenaAllocator.init(allocator);
            errdefer tc.arena.deinit();
            const aa = tc.arena.allocator();

            var bundle = std.crypto.Certificate.Bundle{};
            try bundle.rescan(aa);

            const buf_len = tls.max_ciphertext_record_len;
            const buf = try aa.alloc(u8, buf_len * 4);

            tc.stream_writer = tcp.writer(buf[0..buf_len][0..buf_len]);
            tc.stream_reader = tcp.reader(buf[buf_len .. 2 * buf_len][0..buf_len]);

            tc.client = try tls.Client.init(
                tc.stream_reader.interface(),
                &tc.stream_writer.interface,
                .{
                    .ca = .{ .bundle = bundle },
                    .host = .{ .explicit = host },
                    .read_buffer = buf[2 * buf_len .. 3 * buf_len][0..buf_len],
                    .write_buffer = buf[3 * buf_len .. 4 * buf_len][0..buf_len],
                },
            );

            self.tls_client = tc;
        }

        // Do WebSocket handshake
        try self.doHandshake(host, path);

        return self;
    }

    fn doHandshake(self: *WsClient, host: []const u8, path: []const u8) !void {
        // Generate random key
        var key_raw: [16]u8 = undefined;
        std.crypto.random.bytes(&key_raw);
        var key_buf: [24]u8 = undefined;
        const ws_key = std.base64.standard.Encoder.encode(&key_buf, &key_raw);

        // Build HTTP upgrade request
        var req_buf: [4096]u8 = undefined;
        const req = std.fmt.bufPrint(&req_buf,
            "GET {s} HTTP/1.1\r\n" ++
            "Host: {s}\r\n" ++
            "Upgrade: websocket\r\n" ++
            "Connection: Upgrade\r\n" ++
            "Sec-WebSocket-Key: {s}\r\n" ++
            "Sec-WebSocket-Version: 13\r\n" ++
            "\r\n",
            .{ path, host, ws_key },
        ) catch return error.PathTooLong;

        std.debug.print("Sending handshake ({d} bytes)...\n", .{req.len});
        try self.rawWrite(req);

        // Read response
        var resp_buf: [4096]u8 = undefined;
        var resp_len: usize = 0;

        while (resp_len < resp_buf.len) {
            const n = try self.rawRead(resp_buf[resp_len..]);
            if (n == 0) return error.ConnectionClosed;
            resp_len += n;

            // Check if we have the full headers (double CRLF)
            if (std.mem.indexOf(u8, resp_buf[0..resp_len], "\r\n\r\n")) |header_end| {
                const response = resp_buf[0 .. header_end + 4];
                std.debug.print("Server response:\n{s}\n", .{response});

                // Check for 101
                if (!std.mem.startsWith(u8, response, "HTTP/1.1 101")) {
                    const first_line_end = std.mem.indexOf(u8, response, "\r\n") orelse response.len;
                    std.debug.print("ERROR: Expected 101, got: {s}\n", .{response[0..first_line_end]});
                    return error.HandshakeFailed;
                }

                // Save any over-read data
                const extra = resp_len - (header_end + 4);
                if (extra > 0) {
                    @memcpy(self.recv_buf[0..extra], resp_buf[header_end + 4 .. resp_len]);
                    self.recv_len = extra;
                }

                std.debug.print("WebSocket handshake OK!\n", .{});
                return;
            }
        }
        return error.ResponseTooLarge;
    }

    fn rawWrite(self: *WsClient, data: []const u8) !void {
        if (self.tls_client) |tc| {
            try tc.client.writer.writeAll(data);
            try tc.client.writer.flush();
            try tc.stream_writer.interface.flush();
        } else if (is_windows) {
            const ws2 = std.os.windows.ws2_32;
            var sent: usize = 0;
            while (sent < data.len) {
                const chunk: i32 = @intCast(@min(data.len - sent, 0x7FFFFFFF));
                const result = ws2.send(self.stream.handle, data.ptr + sent, chunk, 0);
                if (result == ws2.SOCKET_ERROR) return error.SendFailed;
                sent += @intCast(result);
            }
        } else {
            try self.stream.writeAll(data);
        }
    }

    fn rawRead(self: *WsClient, buf: []u8) !usize {
        if (buf.len == 0) return 0;
        if (self.tls_client) |tc| {
            var w: std.Io.Writer = .fixed(buf);
            while (true) {
                const n = try tc.client.reader.stream(&w, .limited(buf.len));
                if (n != 0) return n;
            }
        }
        if (is_windows) {
            const ws2 = std.os.windows.ws2_32;
            const len: i32 = @intCast(@min(buf.len, 0x7FFFFFFF));
            const result = ws2.recv(self.stream.handle, buf.ptr, len, 0);
            if (result == ws2.SOCKET_ERROR) return error.RecvFailed;
            return @intCast(result);
        }
        return self.stream.read(buf);
    }

    // Separate buffer for payload to avoid corruption during shift
    var payload_buf: [512 * 1024]u8 = undefined;

    pub fn recvMessage(self: *WsClient) ![]const u8 {
        while (true) {
            // Read WebSocket frame header
            while (self.recv_len < 2) {
                const n = try self.rawRead(self.recv_buf[self.recv_len..]);
                if (n == 0) return error.ConnectionClosed;
                self.recv_len += n;
            }

            const b0 = self.recv_buf[0];
            const b1 = self.recv_buf[1];
            const opcode = b0 & 0x0F;
            const masked = (b1 & 0x80) != 0;
            var payload_len: u64 = b1 & 0x7F;
            var header_len: usize = 2;

            if (payload_len == 126) {
                while (self.recv_len < 4) {
                    const n = try self.rawRead(self.recv_buf[self.recv_len..]);
                    if (n == 0) return error.ConnectionClosed;
                    self.recv_len += n;
                }
                payload_len = @as(u64, self.recv_buf[2]) << 8 | @as(u64, self.recv_buf[3]);
                header_len = 4;
            } else if (payload_len == 127) {
                while (self.recv_len < 10) {
                    const n = try self.rawRead(self.recv_buf[self.recv_len..]);
                    if (n == 0) return error.ConnectionClosed;
                    self.recv_len += n;
                }
                payload_len = 0;
                for (0..8) |i| {
                    payload_len = (payload_len << 8) | @as(u64, self.recv_buf[2 + i]);
                }
                header_len = 10;
            }

            if (masked) header_len += 4;

            const plen: usize = @intCast(payload_len);
            const total: usize = header_len + plen;

            if (total > self.recv_buf.len) {
                std.debug.print("Message too large: {d} bytes (buffer: {d})\n", .{ total, self.recv_buf.len });
                return error.MessageTooLarge;
            }

            // Read until we have the full frame
            while (self.recv_len < total) {
                const n = try self.rawRead(self.recv_buf[self.recv_len..]);
                if (n == 0) return error.ConnectionClosed;
                self.recv_len += n;
            }

            // Unmask if needed
            if (masked) {
                const mask_start = header_len - 4;
                const mask = self.recv_buf[mask_start..][0..4];
                for (self.recv_buf[header_len..total], 0..) |*b, i| {
                    b.* ^= mask[i % 4];
                }
            }

            // Handle control frames (ping, pong, close)
            if (opcode == 0x9) {
                // Ping → respond with pong (same payload)
                self.sendPong(self.recv_buf[header_len..total]) catch {};
                // Consume frame and continue reading
                const remaining = self.recv_len - total;
                if (remaining > 0) {
                    std.mem.copyForwards(u8, self.recv_buf[0..remaining], self.recv_buf[total..self.recv_len]);
                }
                self.recv_len = remaining;
                continue; // Read next frame
            }

            if (opcode == 0x8) {
                // Close frame
                std.debug.print("Server sent close frame\n", .{});
                return error.ConnectionClosed;
            }

            if (opcode == 0xA) {
                // Pong → ignore
                const remaining = self.recv_len - total;
                if (remaining > 0) {
                    std.mem.copyForwards(u8, self.recv_buf[0..remaining], self.recv_buf[total..self.recv_len]);
                }
                self.recv_len = remaining;
                continue;
            }

            // Text (1) or Binary (2) data frame → return payload
            if (plen > payload_buf.len) return error.MessageTooLarge;
            @memcpy(payload_buf[0..plen], self.recv_buf[header_len..total]);

            // Shift remaining data
            const remaining = self.recv_len - total;
            if (remaining > 0) {
                std.mem.copyForwards(u8, self.recv_buf[0..remaining], self.recv_buf[total..self.recv_len]);
            }
            self.recv_len = remaining;

            return payload_buf[0..plen];
        }
    }

    fn sendPong(self: *WsClient, payload: []const u8) !void {
        var frame_buf: [256]u8 = undefined;
        var pos: usize = 0;
        // FIN + pong opcode (0x8A)
        frame_buf[0] = 0x8A;
        pos = 1;
        // Masked, length
        if (payload.len < 126) {
            frame_buf[1] = @intCast(payload.len | 0x80);
            pos = 2;
        } else {
            // Pong payloads should be small
            return;
        }
        // Mask key
        var mask: [4]u8 = undefined;
        std.crypto.random.bytes(&mask);
        @memcpy(frame_buf[pos .. pos + 4], &mask);
        pos += 4;
        // Masked payload
        for (payload, 0..) |b, i| {
            frame_buf[pos + i] = b ^ mask[i % 4];
        }
        pos += payload.len;
        try self.rawWrite(frame_buf[0..pos]);
    }

    pub fn sendMessage(self: *WsClient, data: []const u8) !void {
        // Build masked text frame
        var frame_buf: [16384]u8 = undefined;
        var pos: usize = 0;

        // FIN + text opcode
        frame_buf[0] = 0x81;
        pos = 1;

        // Length + mask bit
        if (data.len < 126) {
            frame_buf[1] = @intCast(data.len | 0x80);
            pos = 2;
        } else if (data.len < 65536) {
            frame_buf[1] = 126 | 0x80;
            frame_buf[2] = @intCast((data.len >> 8) & 0xFF);
            frame_buf[3] = @intCast(data.len & 0xFF);
            pos = 4;
        } else {
            return error.MessageTooLarge;
        }

        // Mask key
        var mask: [4]u8 = undefined;
        std.crypto.random.bytes(&mask);
        @memcpy(frame_buf[pos .. pos + 4], &mask);
        pos += 4;

        // Masked payload
        for (data, 0..) |b, i| {
            frame_buf[pos + i] = b ^ mask[i % 4];
        }
        pos += data.len;

        try self.rawWrite(frame_buf[0..pos]);
    }

    pub fn deinit(self: *WsClient) void {
        if (self.tls_client) |tc| {
            _ = tc.client.end() catch {};
            tc.arena.deinit();
            self.allocator.destroy(tc);
        }
        self.stream.close();
        self.allocator.free(self.recv_buf);
    }
};
