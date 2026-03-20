# Git Statistics Script - Since Repository Start (Per Author Breakdown)
# Run this script from your repository root

$ErrorActionPreference = "SilentlyContinue"

# =================================================================
# AUTHOR ALIAS MAPPING
# Map multiple identities to a single canonical name
# =================================================================
$authorAliases = @{
    "dag.thomas.olsen@simployer.com" = "Dag Thomas Olsen"
    "dagthomas@gmail.com"            = "Dag Thomas Olsen"
    "Lars Henrik Mostad Haugeli"     = "Lars Haugeli"
    "tomaszjaworski"                 = "Tomasz Jaworski"
}

function Resolve-Author($name, $email) {
    if ($email -and $authorAliases.ContainsKey($email)) {
        return $authorAliases[$email]
    }
    if ($name -and $authorAliases.ContainsKey($name)) {
        return $authorAliases[$name]
    }
    return $name
}

# Colors for output
function Write-Header($text) {
    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Cyan
    Write-Host "  $text" -ForegroundColor Yellow
    Write-Host "==================================================================" -ForegroundColor Cyan
}

function Write-SubHeader($text) {
    Write-Host ""
    Write-Host "  ----------------------------------------------------------------" -ForegroundColor DarkCyan
    Write-Host "    $text" -ForegroundColor White
    Write-Host "  ----------------------------------------------------------------" -ForegroundColor DarkCyan
}

function Write-Stat($label, $value) {
    Write-Host "    ${label}: " -NoNewline -ForegroundColor Gray
    Write-Host "$value" -ForegroundColor Green
}

# Check if we're in a git repository
$gitCheck = git rev-parse --is-inside-work-tree 2>&1
if ($gitCheck -ne "true") {
    Write-Host "Error: Not a git repository!" -ForegroundColor Red
    exit 1
}

$repoName = Split-Path -Leaf (git rev-parse --show-toplevel)
$repoStart = git log --reverse --format="%ai" 2>$null | Select-Object -First 1

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Magenta
Write-Host "       GIT STATISTICS - ALL TIME (PER AUTHOR)                     " -ForegroundColor Magenta
Write-Host "       Repository: $repoName" -ForegroundColor Magenta
if ($repoStart) {
    Write-Host "       Since: $repoStart" -ForegroundColor Magenta
}
Write-Host "==================================================================" -ForegroundColor Magenta

# =================================================================
# OVERALL SUMMARY
# =================================================================
Write-Header "OVERALL SUMMARY"

$totalCommits = (git rev-list --count HEAD 2>$null)
if (-not $totalCommits) { $totalCommits = 0 }
Write-Stat "Total Commits" $totalCommits

$firstCommit = git log --format="%ai" --reverse 2>$null | Select-Object -First 1
$lastCommit = git log --format="%ai" 2>$null | Select-Object -First 1
if ($firstCommit) { Write-Stat "First Commit" $firstCommit }
if ($lastCommit) { Write-Stat "Last Commit" $lastCommit }

# Repo age
if ($firstCommit) {
    $startDate = [datetime]::Parse($firstCommit.Substring(0, 10))
    $endDate = Get-Date
    $repoAge = ($endDate - $startDate).Days
    Write-Stat "Repo Age" "$repoAge days"
}

# =================================================================
# COLLECT ALL COMMIT DATA WITH AUTHOR INFO
# =================================================================

# Get all commits with author name, email, date, hash, subject
$allCommits = git log --format="%H|%ae|%an|%ad|%s" --date=short 2>$null |
    ForEach-Object {
        $parts = $_ -split '\|', 5
        if ($parts.Count -ge 5) {
            $resolved = Resolve-Author $parts[2] $parts[1]
            [PSCustomObject]@{
                Hash    = $parts[0]
                Email   = $parts[1]
                RawName = $parts[2]
                Author  = $resolved
                Date    = $parts[3]
                Subject = $parts[4]
            }
        }
    }

# Get unique authors
$uniqueAuthors = $allCommits | Group-Object Author | Sort-Object Count -Descending

Write-Stat "Contributors" $uniqueAuthors.Count

# Pre-compute per-author line stats for the summary table
$authorSummaries = @{}
foreach ($authorGroup in $uniqueAuthors) {
    $emails = $authorGroup.Group | Select-Object -ExpandProperty Email | Sort-Object -Unique
    $aStats = @()
    foreach ($email in $emails) {
        $lines = git log --author="$email" --pretty=tformat: --numstat 2>$null |
            Where-Object { $_ -match '^\d' } |
            ForEach-Object {
                $p = $_ -split '\t'
                [PSCustomObject]@{ Added = [int]$p[0]; Deleted = [int]$p[1] }
            }
        if ($lines) { $aStats += $lines }
    }
    $aAdded = ($aStats | Measure-Object -Property Added -Sum).Sum
    $aDeleted = ($aStats | Measure-Object -Property Deleted -Sum).Sum
    if (-not $aAdded) { $aAdded = 0 }
    if (-not $aDeleted) { $aDeleted = 0 }

    $aFiles = @()
    foreach ($email in $emails) {
        $f = git log --author="$email" --name-only --pretty=format: 2>$null |
            Where-Object { $_ }
        if ($f) { $aFiles += $f }
    }
    $aUniqueFiles = ($aFiles | Sort-Object -Unique).Count

    $aDays = ($authorGroup.Group | Select-Object -ExpandProperty Date | Sort-Object -Unique).Count

    $authorSummaries[$authorGroup.Name] = @{
        Commits = $authorGroup.Count
        Added   = $aAdded
        Deleted = $aDeleted
        Net     = $aAdded - $aDeleted
        Files   = $aUniqueFiles
        Days    = $aDays
    }
}

Write-Host ""
Write-Host "    Author                              Commits   +Added   -Deleted      Net   Files  Days" -ForegroundColor Gray
Write-Host "    ------                              -------   ------   --------      ---   -----  ----" -ForegroundColor DarkGray
foreach ($authorGroup in $uniqueAuthors) {
    $s = $authorSummaries[$authorGroup.Name]
    $namePad   = $authorGroup.Name.PadRight(40)
    $commits   = $s.Commits.ToString().PadLeft(7)
    $addedStr  = ("+$($s.Added)").PadLeft(8)
    $deletedStr = ("-$($s.Deleted)").PadLeft(10)
    $netStr    = $(if ($s.Net -ge 0) { "+$($s.Net)" } else { "$($s.Net)" }).PadLeft(8)
    $filesStr  = $s.Files.ToString().PadLeft(7)
    $daysStr   = $s.Days.ToString().PadLeft(5)
    Write-Host "    $namePad" -NoNewline -ForegroundColor Cyan
    Write-Host "$commits" -NoNewline -ForegroundColor White
    Write-Host "$addedStr" -NoNewline -ForegroundColor Green
    Write-Host "$deletedStr" -NoNewline -ForegroundColor Red
    if ($s.Net -ge 0) {
        Write-Host "$netStr" -NoNewline -ForegroundColor Green
    } else {
        Write-Host "$netStr" -NoNewline -ForegroundColor Red
    }
    Write-Host "$filesStr" -NoNewline -ForegroundColor White
    Write-Host "$daysStr" -ForegroundColor White
}

# =================================================================
# PER-AUTHOR BREAKDOWN
# =================================================================

foreach ($authorGroup in $uniqueAuthors) {
    $authorName = $authorGroup.Name
    $authorCommits = $authorGroup.Group

    Write-Header "AUTHOR: $authorName"

    # --- Commit Stats ---
    Write-SubHeader "Commits"
    Write-Stat "Total Commits" $authorCommits.Count

    $authorFirstCommit = $authorCommits | Sort-Object Date | Select-Object -First 1
    $authorLastCommit = $authorCommits | Sort-Object Date -Descending | Select-Object -First 1
    if ($authorFirstCommit) { Write-Stat "First Commit" $authorFirstCommit.Date }
    if ($authorLastCommit) { Write-Stat "Last Commit" $authorLastCommit.Date }

    # Active days
    $activeDays = $authorCommits | Select-Object -ExpandProperty Date | Sort-Object -Unique
    Write-Stat "Active Days" $activeDays.Count

    # Avg commits per active day
    if ($activeDays.Count -gt 0) {
        $avgPerDay = [Math]::Round($authorCommits.Count / $activeDays.Count, 1)
        Write-Stat "Avg Commits/Active Day" $avgPerDay
    }

    # --- Commits by Month ---
    Write-SubHeader "Commits by Month (Top 15)"
    $commitsByMonth = $authorCommits |
        ForEach-Object { [PSCustomObject]@{ Month = $_.Date.Substring(0, 7); Hash = $_.Hash } } |
        Group-Object Month |
        Sort-Object Name -Descending |
        Select-Object -First 15
    if ($commitsByMonth) {
        foreach ($month in ($commitsByMonth | Sort-Object Name)) {
            $barLength = [Math]::Min($month.Count, 50)
            $bar = "#" * $barLength
            Write-Host "    $($month.Name) " -NoNewline -ForegroundColor Cyan
            Write-Host "$bar " -NoNewline -ForegroundColor Green
            Write-Host "($($month.Count))" -ForegroundColor White
        }
    }

    # --- Line Changes ---
    Write-SubHeader "Line Changes"

    # Collect all emails/names for this resolved author
    $authorEmails = $authorCommits | Select-Object -ExpandProperty Email | Sort-Object -Unique

    $authorStats = @()
    foreach ($email in $authorEmails) {
        $lines = git log --author="$email" --pretty=tformat: --numstat 2>$null |
            Where-Object { $_ -match '^\d' } |
            ForEach-Object {
                $parts = $_ -split '\t'
                [PSCustomObject]@{
                    Added   = [int]$parts[0]
                    Deleted = [int]$parts[1]
                    File    = $parts[2]
                }
            }
        if ($lines) { $authorStats += $lines }
    }

    $added = ($authorStats | Measure-Object -Property Added -Sum).Sum
    $deleted = ($authorStats | Measure-Object -Property Deleted -Sum).Sum
    if (-not $added) { $added = 0 }
    if (-not $deleted) { $deleted = 0 }
    $net = $added - $deleted

    Write-Stat "Lines Added" "+$added"
    Write-Stat "Lines Deleted" "-$deleted"
    Write-Host "    Net Change: " -NoNewline -ForegroundColor Gray
    if ($net -ge 0) {
        Write-Host "+$net" -ForegroundColor Green
    } else {
        Write-Host "$net" -ForegroundColor Red
    }

    # --- File Changes ---
    Write-SubHeader "File Changes"

    $authorFiles = @()
    foreach ($email in $authorEmails) {
        $files = git log --author="$email" --name-only --pretty=format: 2>$null |
            Where-Object { $_ }
        if ($files) { $authorFiles += $files }
    }

    $uniqueFiles = $authorFiles | Sort-Object -Unique
    Write-Stat "Unique Files Modified" $uniqueFiles.Count

    # Top file types
    $extensions = $uniqueFiles |
        ForEach-Object { [System.IO.Path]::GetExtension($_) } |
        Where-Object { $_ } |
        Group-Object |
        Sort-Object Count -Descending |
        Select-Object -First 10

    if ($extensions) {
        Write-Host ""
        Write-Host "    Top File Types:" -ForegroundColor Gray
        foreach ($ext in $extensions) {
            $extName = $ext.Name.PadRight(15)
            Write-Host "      $extName " -NoNewline -ForegroundColor Cyan
            Write-Host "$($ext.Count) files" -ForegroundColor White
        }
    }

    # --- Most Modified Files ---
    Write-SubHeader "Most Modified Files (Top 10)"

    $activeFiles = $authorFiles |
        Group-Object |
        Sort-Object Count -Descending |
        Select-Object -First 10

    if ($activeFiles) {
        foreach ($file in $activeFiles) {
            $fileName = $file.Name
            if ($fileName.Length -gt 50) {
                $fileName = "..." + $fileName.Substring($fileName.Length - 47)
            }
            $countPad = $file.Count.ToString().PadLeft(3)
            Write-Host "      $countPad changes: " -NoNewline -ForegroundColor Yellow
            Write-Host $fileName -ForegroundColor White
        }
    } else {
        Write-Host "    No files found" -ForegroundColor Gray
    }

    # --- Recent Commits ---
    Write-SubHeader "Recent Commits (Last 10)"

    $recentAuthorCommits = $authorCommits |
        Sort-Object { $_.Hash } |
        Select-Object -Last 10

    # Re-fetch with short hash for display
    $recentHashes = $recentAuthorCommits | Select-Object -ExpandProperty Hash
    foreach ($hash in $recentHashes) {
        $shortHash = $hash.Substring(0, 7)
        $commit = $authorCommits | Where-Object { $_.Hash -eq $hash } | Select-Object -First 1
        if ($commit) {
            $message = $commit.Subject
            if ($message.Length -gt 50) {
                $message = $message.Substring(0, 47) + "..."
            }
            Write-Host "    $shortHash " -NoNewline -ForegroundColor Yellow
            Write-Host "$($commit.Date) " -NoNewline -ForegroundColor Cyan
            Write-Host "$message" -ForegroundColor White
        }
    }
}

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
$reportDate = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
Write-Host "  Report generated: $reportDate" -ForegroundColor Gray
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""
