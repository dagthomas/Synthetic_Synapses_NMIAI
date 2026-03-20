// Reactive solver with trip-based bot assignment.
// Each bot gets a full trip (up to 3 items → dropoff).
// BFS navigation, exact game simulation, no spacetime needed.

#include <cstdio>
#include <cstring>
#include <vector>
#include <queue>
#include <algorithm>
#include <numeric>

using namespace std;

static const int MAXW=32, MAXH=20, MAXCELLS=MAXW*MAXH, MAXBOTS=22;
static const int MAXROUNDS=502, INV_CAP=3, MAX_ORDERS=200, INF=1e8;
static const int ACT_WAIT=0, ACT_UP=1, ACT_DOWN=2, ACT_LEFT=3, ACT_RIGHT=4, ACT_PICKUP=5, ACT_DROPOFF=6;
static const int DX[]={0,0,0,-1,1,0,0}, DY[]={0,-1,1,0,0,0,0};

struct Vec2 { int x,y; bool operator==(const Vec2& o) const {return x==o.x&&y==o.y;} };
struct Item { int type_id; Vec2 pos; vector<Vec2> adj; };
struct Order { vector<int> types; };
struct Action { int act, arg; };

int W,H,num_bots,max_rounds,num_types,one_way_row=-1;
int grid[MAXW][MAXH];
Vec2 spawn_pos;
vector<Vec2> dropoff_zones;
vector<Item> items;
vector<Order> orders;
int bfs_dist[MAXCELLS][MAXCELLS];
vector<vector<int>> type_to_items;
static Action action_log[MAXROUNDS][MAXBOTS];

inline int cidx(Vec2 p){return p.y*W+p.x;}
inline bool walkable(int x,int y){return x>=0&&x<W&&y>=0&&y<H&&(grid[x][y]==0||grid[x][y]==3);}
bool is_dropoff(Vec2 p){for(auto&d:dropoff_zones)if(d==p)return true;return false;}
bool is_spawn(Vec2 p){return p==spawn_pos;}
int get_dist(Vec2 a,Vec2 b){return bfs_dist[cidx(a)][cidx(b)];}

void read_input(){
    scanf("%d%d%d%d",&W,&H,&num_bots,&max_rounds);
    int nd;scanf("%d",&nd);dropoff_zones.resize(nd);
    for(int i=0;i<nd;i++)scanf("%d%d",&dropoff_zones[i].x,&dropoff_zones[i].y);
    scanf("%d%d",&spawn_pos.x,&spawn_pos.y);
    char line[64];
    for(int y=0;y<H;y++){scanf("%s",line);for(int x=0;x<W;x++)
        grid[x][y]=(line[x]=='#')?1:(line[x]=='S')?2:(line[x]=='D')?3:0;}
    int ni;scanf("%d",&ni);items.resize(ni);
    for(int i=0;i<ni;i++)scanf("%d%d%d",&items[i].type_id,&items[i].pos.x,&items[i].pos.y);
    scanf("%d",&num_types);type_to_items.resize(num_types);
    for(int i=0;i<ni;i++)type_to_items[items[i].type_id].push_back(i);
    int no;scanf("%d",&no);orders.resize(no);
    for(int i=0;i<no;i++){int n;scanf("%d",&n);orders[i].types.resize(n);for(int j=0;j<n;j++)scanf("%d",&orders[i].types[j]);}
    scanf("%d",&one_way_row);
    for(int i=0;i<ni;i++){int ix=items[i].pos.x,iy=items[i].pos.y;
        for(int d=1;d<=4;d++){int nx=ix+DX[d],ny=iy+DY[d];if(walkable(nx,ny))items[i].adj.push_back({nx,ny});}}
}

void precompute_bfs(){
    for(int y=0;y<H;y++)for(int x=0;x<W;x++)if(walkable(x,y)){
        int si=cidx({x,y});for(int i=0;i<W*H;i++)bfs_dist[si][i]=INF;bfs_dist[si][si]=0;
        queue<Vec2>q;q.push({x,y});while(!q.empty()){Vec2 c=q.front();q.pop();int ci=cidx(c);
            for(int d=1;d<=4;d++){int nx=c.x+DX[d],ny=c.y+DY[d];if(!walkable(nx,ny))continue;
                int ni=cidx({nx,ny});if(bfs_dist[si][ni]!=INF)continue;bfs_dist[si][ni]=bfs_dist[si][ci]+1;q.push({nx,ny});}}
    }
}

int nearest_item_of_type(int tid,Vec2 ref){
    int best=INF,bi=-1;
    for(int idx:type_to_items[tid])for(auto&a:items[idx].adj){int d=get_dist(ref,a);if(d<best){best=d;bi=idx;}}
    return bi;
}

// ---- Simulation (exact match of game_engine.py) ----
struct SimState {
    Vec2 pos[MAXBOTS]; int8_t inv[MAXBOTS][INV_CAP]; int inv_count[MAXBOTS];
    int active_order,next_order; bool delivered[MAX_ORDERS][8]; bool complete[MAX_ORDERS];
    int score,items_delivered,orders_completed;
    void init(){for(int b=0;b<num_bots;b++){pos[b]=spawn_pos;memset(inv[b],-1,sizeof(inv[b]));inv_count[b]=0;}
        active_order=0;next_order=2;memset(delivered,0,sizeof(delivered));memset(complete,0,sizeof(complete));score=0;items_delivered=0;orders_completed=0;}
    bool inv_add(int b,int t){for(int i=0;i<INV_CAP;i++)if(inv[b][i]==-1){inv[b][i]=t;inv_count[b]++;return true;}return false;}
    int deliver_matching(int b,int oi){
        int cnt=0;int8_t keep[INV_CAP];int ki=0;
        for(int i=0;i<INV_CAP;i++){if(inv[b][i]<0)continue;int t=inv[b][i];bool m=false;
            for(int j=0;j<(int)orders[oi].types.size();j++)if(orders[oi].types[j]==t&&!delivered[oi][j]){delivered[oi][j]=true;m=true;cnt++;break;}
            if(!m)keep[ki++]=t;}
        memset(inv[b],-1,sizeof(inv[b]));for(int i=0;i<ki;i++)inv[b][i]=keep[i];inv_count[b]=ki;return cnt;}
    bool order_done(int oi){for(int j=0;j<(int)orders[oi].types.size();j++)if(!delivered[oi][j])return false;return true;}
};

void sim_step(SimState&s,Action acts[]){
    int occ[MAXW][MAXH];memset(occ,0,sizeof(occ));
    for(int b=0;b<num_bots;b++)occ[s.pos[b].x][s.pos[b].y]++;
    for(int bid=0;bid<num_bots;bid++){
        int act=acts[bid].act,arg=acts[bid].arg,bx=s.pos[bid].x,by=s.pos[bid].y;
        if(act>=1&&act<=4){int nx=bx+DX[act],ny=by+DY[act];
            if(walkable(nx,ny)&&(occ[nx][ny]==0||is_spawn({nx,ny}))){occ[bx][by]--;s.pos[bid]={nx,ny};occ[nx][ny]++;}}
        else if(act==ACT_PICKUP&&arg>=0&&arg<(int)items.size()&&s.inv_count[bid]<INV_CAP){
            if(abs(bx-items[arg].pos.x)+abs(by-items[arg].pos.y)==1)s.inv_add(bid,items[arg].type_id);}
        else if(act==ACT_DROPOFF&&is_dropoff({bx,by})&&s.inv_count[bid]>0){
            int oi=s.active_order;if(oi<(int)orders.size()&&!s.complete[oi]){
                int d=s.deliver_matching(bid,oi);s.score+=d;s.items_delivered+=d;
                while(oi<(int)orders.size()&&s.order_done(oi)){s.complete[oi]=true;s.score+=5;s.orders_completed++;
                    s.active_order=oi+1;if(s.next_order<(int)orders.size())s.next_order++;oi=s.active_order;
                    if(oi>=(int)orders.size())break;
                    for(int b2=0;b2<num_bots;b2++)if(is_dropoff(s.pos[b2])){
                        int d2=s.deliver_matching(b2,oi);s.score+=d2;s.items_delivered+=d2;}}}}
    }
}

// ---- Trip-based reactive solver ----
// Each bot has a trip: sequence of items to pick up, then deliver

struct BotTrip {
    vector<int> items_to_pickup; // item indices, in order
    int pickup_idx;              // next item to pick up (index into items_to_pickup)
    bool delivering;             // true when all picked up, heading to dropoff
    bool done;                   // trip complete
    int order_idx;               // which order this trip serves
};

// DP grouping of items into trips of ≤3
vector<vector<int>> group_items_dp(vector<int>& chosen, Vec2 ref) {
    int n = chosen.size();
    if (n == 0) return {};
    if (n <= 3) {
        // Try all permutations for best order
        vector<int> perm(n); iota(perm.begin(),perm.end(),0);
        int best=INF; vector<int> best_ord;
        do {
            int cost=0; Vec2 p=ref;
            for(int i=0;i<n;i++){int idx=chosen[perm[i]];
                int bd=INF;for(auto&a:items[idx].adj){int d=get_dist(p,a);if(d<bd)bd=d;}
                cost+=bd;if(!items[idx].adj.empty()){int bd2=INF;Vec2 ba=items[idx].adj[0];
                    for(auto&a:items[idx].adj){int d=get_dist(p,a);if(d<bd2){bd2=d;ba=a;}}p=ba;}}
            if(cost<best){best=cost;best_ord.clear();for(int i=0;i<n;i++)best_ord.push_back(chosen[perm[i]]);}
        } while(next_permutation(perm.begin(),perm.end()));
        return {best_ord};
    }
    // DP bitmask grouping
    int full=(1<<n)-1;
    vector<int> dp(full+1,INF),dp_par(full+1,-1),dp_sub(full+1,-1);
    dp[0]=0;
    for(int mask=0;mask<=full;mask++){if(dp[mask]==INF)continue;int remain=full&~mask;
        for(int sub=remain;sub>0;sub=(sub-1)&remain){if(__builtin_popcount(sub)>3)continue;
            vector<int>ti;for(int i=0;i<n;i++)if(sub&(1<<i))ti.push_back(chosen[i]);
            // Estimate cost
            int cost=0;Vec2 p=ref;for(int idx:ti){int bd=INF;
                for(auto&a:items[idx].adj){int d=get_dist(p,a);if(d<bd)bd=d;}cost+=bd;
                if(!items[idx].adj.empty()){int bd2=INF;Vec2 ba=items[idx].adj[0];
                    for(auto&a:items[idx].adj){int d=get_dist(p,a);if(d<bd2){bd2=d;ba=a;}}p=ba;}}
            int nm=mask|sub;if(dp[mask]+cost<dp[nm]){dp[nm]=dp[mask]+cost;dp_par[nm]=mask;dp_sub[nm]=sub;}}}
    vector<vector<int>> result;
    int cur=full;while(cur!=0){int sub=dp_sub[cur];
        vector<int>ti;for(int i=0;i<n;i++)if(sub&(1<<i))ti.push_back(chosen[i]);result.push_back(ti);cur=dp_par[cur];}
    return result;
}

int bfs_next_toward(Vec2 from, Vec2 goal, int occ[][MAXH]) {
    if (from == goal) return ACT_WAIT;
    int best_act=ACT_WAIT, best_dist=INF;
    for(int d=1;d<=4;d++){
        int nx=from.x+DX[d],ny=from.y+DY[d];
        if(!walkable(nx,ny))continue;
        if(occ[nx][ny]>0 && !is_spawn({nx,ny}))continue;
        int dist=bfs_dist[cidx({nx,ny})][cidx(goal)];
        if(dist<best_dist){best_dist=dist;best_act=d;}
    }
    return best_act;
}

void solve() {
    fprintf(stderr,"Reactive solver: %d bots, %d orders, %d rounds\n",num_bots,(int)orders.size(),max_rounds);

    SimState sim; sim.init();
    BotTrip trips[MAXBOTS];
    for(int b=0;b<num_bots;b++){trips[b].done=true;trips[b].delivering=false;trips[b].pickup_idx=0;trips[b].order_idx=-1;}

    int next_order_to_assign = 0; // next order to create trips for

    for(int r=0;r<max_rounds;r++){
        int oi = sim.active_order;

        // Keep next_order_to_assign at least at active order
        if (next_order_to_assign < oi) next_order_to_assign = oi;

        // Assign trips to idle bots
        while(next_order_to_assign < (int)orders.size()) {
            // Check if we have idle bots
            int idle_count = 0;
            for(int b=0;b<num_bots;b++) if(trips[b].done) idle_count++;
            if(idle_count == 0) break;

            int assign_oi = next_order_to_assign;
            Vec2 ref = dropoff_zones[0];

            // Choose items for this order
            vector<int> chosen;
            for(int tid : orders[assign_oi].types){
                int best = nearest_item_of_type(tid, ref);
                if(best>=0) chosen.push_back(best);
            }
            if(chosen.empty()){next_order_to_assign++;continue;}

            // Group into trips
            auto trip_groups = group_items_dp(chosen, ref);
            if(trip_groups.empty()){next_order_to_assign++;continue;}

            // Assign trips to nearest idle bots
            bool assigned_any = false;
            bool all_assigned = true;
            for(auto& tg : trip_groups){
                int best_b=-1, best_d=INF;
                for(int b=0;b<num_bots;b++){
                    if(!trips[b].done) continue;
                    int d = 0;
                    if(!tg.empty() && !items[tg[0]].adj.empty())
                        d = get_dist(sim.pos[b], items[tg[0]].adj[0]);
                    if(d<best_d){best_d=d;best_b=b;}
                }
                if(best_b>=0){
                    trips[best_b].items_to_pickup = tg;
                    trips[best_b].pickup_idx = 0;
                    trips[best_b].delivering = false;
                    trips[best_b].done = false;
                    trips[best_b].order_idx = assign_oi;
                    assigned_any = true;
                } else {
                    all_assigned = false;
                }
            }
            // Only advance if ALL trips were assigned
            if(all_assigned) next_order_to_assign++;
            else break; // retry next round when bots free up
            if(!assigned_any) break;
            // Don't assign too far ahead (max 3 orders ahead of active)
            if(next_order_to_assign > oi + 3) break;
        }

        // Build occupied count
        int occ[MAXW][MAXH]; memset(occ,0,sizeof(occ));
        for(int b=0;b<num_bots;b++) occ[sim.pos[b].x][sim.pos[b].y]++;

        // Decide actions for each bot
        Action acts[MAXBOTS];
        for(int b=0;b<num_bots;b++){
            Vec2 bp = sim.pos[b];
            BotTrip& trip = trips[b];

            if(trip.done){
                // Idle: go to spawn
                if(bp==spawn_pos) acts[b]={ACT_WAIT,-1};
                else acts[b]={bfs_next_toward(bp,spawn_pos,occ),-1};
            } else if(trip.delivering) {
                // Navigate to dropoff and deliver
                if(is_dropoff(bp)){
                    acts[b]={ACT_DROPOFF,-1};
                    trip.done = true;
                } else {
                    Vec2 dz=dropoff_zones[0];int bd=INF;
                    for(auto&d:dropoff_zones){int dd=get_dist(bp,d);if(dd<bd){bd=dd;dz=d;}}
                    acts[b]={bfs_next_toward(bp,dz,occ),-1};
                }
            } else {
                // Picking up items
                if(trip.pickup_idx >= (int)trip.items_to_pickup.size()){
                    // All items picked up, switch to delivering
                    trip.delivering = true;
                    if(is_dropoff(bp)){
                        acts[b]={ACT_DROPOFF,-1};
                        trip.done = true;
                    } else {
                        Vec2 dz=dropoff_zones[0];int bd=INF;
                        for(auto&d:dropoff_zones){int dd=get_dist(bp,d);if(dd<bd){bd=dd;dz=d;}}
                        acts[b]={bfs_next_toward(bp,dz,occ),-1};
                    }
                } else {
                    int item_idx = trip.items_to_pickup[trip.pickup_idx];
                    Vec2 ipos = items[item_idx].pos;
                    // Check if adjacent to item
                    if(abs(bp.x-ipos.x)+abs(bp.y-ipos.y)==1){
                        acts[b]={ACT_PICKUP,item_idx};
                        trip.pickup_idx++;
                    } else {
                        // Navigate to best adjacent cell of item
                        Vec2 target={-1,-1}; int bd=INF;
                        for(auto&a:items[item_idx].adj){int d=get_dist(bp,a);if(d<bd){bd=d;target=a;}}
                        if(target.x>=0)
                            acts[b]={bfs_next_toward(bp,target,occ),-1};
                        else
                            acts[b]={ACT_WAIT,-1};
                    }
                }
            }

            // Update occupied for subsequent bots
            Vec2 np=bp;
            if(acts[b].act>=1&&acts[b].act<=4){np={bp.x+DX[acts[b].act],bp.y+DY[acts[b].act]};}
            if(np!=bp){occ[bp.x][bp.y]--;occ[np.x][np.y]++;}
        }

        // Store and simulate
        for(int b=0;b<num_bots;b++) action_log[r][b]=acts[b];
        sim_step(sim,acts);

        if(r%100==0||r==max_rounds-1)
            fprintf(stderr,"  R%d: score=%d orders=%d active=%d\n",r,sim.score,sim.orders_completed,sim.active_order);
    }
    fprintf(stderr,"Final: score=%d orders=%d items=%d\n",sim.score,sim.orders_completed,sim.items_delivered);
}

void output(){printf("%d %d\n",max_rounds,num_bots);
    for(int r=0;r<max_rounds;r++){for(int b=0;b<num_bots;b++){if(b)printf(" ");printf("%d %d",action_log[r][b].act,action_log[r][b].arg);}printf("\n");}}
int main(){read_input();
    fprintf(stderr,"Map: %dx%d, %d bots, %d items, %d types, %d orders\n",W,H,num_bots,(int)items.size(),num_types,(int)orders.size());
    precompute_bfs();fprintf(stderr,"BFS done\n");solve();output();return 0;}
