$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../../..')).Path
$taskOut = Join-Path $taskRoot 'output/pareto-ppo-v9'
$taskProtocol = Get-Content -LiteralPath (Join-Path $taskOut 'protocol.json') -Raw | ConvertFrom-Json
$taskProtocolHash = (Get-FileHash -LiteralPath (Join-Path $taskOut 'protocol.json') -Algorithm SHA256).Hash.ToLowerInvariant()
$taskRunName = 'coload-aux-' + (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$taskRunDir = Join-Path $PSScriptRoot $taskRunName
if (Test-Path -LiteralPath $taskRunDir) { throw 'Refuse to overwrite orchestration run' }
New-Item -ItemType Directory -Path $taskRunDir | Out-Null
$taskManifestPath = Join-Path $taskRunDir 'manifest.json'
$taskRecords = [System.Collections.Generic.List[object]]::new()
$taskRunning = [System.Collections.Generic.List[object]]::new()
$taskIds = @('PPO-coload-70-large-r0','PPO-coload-70-large-r1','PPO-coload-70-large-r2',
    'PPO-coload-80-large-r0','PPO-coload-80-large-r1','PPO-coload-80-large-r2',
    'PPO-coload-120-large-r0','PPO-coload-120-large-r1','PPO-coload-120-large-r2',
    'PPO-coload-130-large-r0','PPO-coload-130-large-r1','PPO-coload-130-large-r2')
$taskPeakAux = 0
function Save-TaskManifest {
    $taskManifest = [ordered]@{
        protocol_sha256 = $taskProtocolHash
        orchestration_source_sha256 = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLowerInvariant()
        updated_utc = (Get-Date).ToUniversalTime().ToString('o')
        task_ids = $taskIds
        auxiliary_worker_count_planned = 12
        existing_main_pool_worker_count = 3
        declared_global_worker_upper_bound = 15
        observed_auxiliary_process_peak = $taskPeakAux
        observed_global_active_solver_peak = $null
        concurrency_note = 'Auxiliary PID count measures live CLI launcher processes, not necessarily active solver tasks. Existing main pool has 3 workers; 15 is declared overlap upper bound, not measured global active-solver peak.'
        timing_note = 'Historical and new concurrent wall-clock times are not isolated speed benchmarks.'
        entries = @($taskRecords.ToArray())
    }
    [System.IO.File]::WriteAllText($taskManifestPath, ($taskManifest | ConvertTo-Json -Depth 20), [System.Text.UTF8Encoding]::new($false))
}
Save-TaskManifest
foreach ($taskId in $taskIds) {
    $taskMatched = @($taskProtocol.tasks | Where-Object { $_.id -eq $taskId })
    if ($taskMatched.Count -ne 1 -or $taskMatched[0].kind -ne 'PPO') { throw "Not a unique frozen PPO task: $taskId" }
    $taskRecord = [ordered]@{ task_id=$taskId; command="py -B run_ppo_frontier_v9.py run --id $taskId --workers 1";
        working_directory=$taskRoot; workers=1; cpu_threads_per_worker=1; status='pending'; pid=$null;
        started_utc=$null; ended_utc=$null; exit_code=$null; stdout="$taskId.stdout.log"; stderr="$taskId.stderr.log" }
    $taskRecords.Add($taskRecord)
    $taskExisting = @("completed/$taskId.json", "actions/$taskId", "progress/$taskId.json") |
        Where-Object { Test-Path -LiteralPath (Join-Path $taskOut $_) }
    if ($taskExisting.Count -gt 0) {
        $taskRecord.status='skipped_existing_start_trace';$taskRecord.existing_paths=@($taskExisting)
        Save-TaskManifest
        continue
    }
    try {
        $taskRecord.started_utc=(Get-Date).ToUniversalTime().ToString('o')
        $taskProcess=Start-Process -FilePath 'py' -ArgumentList @('-B','run_ppo_frontier_v9.py','run','--id',$taskId,'--workers','1') `
            -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $taskRunDir $taskRecord.stdout) `
            -RedirectStandardError (Join-Path $taskRunDir $taskRecord.stderr)
        $taskRecord.pid=$taskProcess.Id;$taskRecord.status='running'
        $taskRunning.Add([pscustomobject]@{ process=$taskProcess; record=$taskRecord })
        $taskPeakAux=[Math]::Max($taskPeakAux,$taskRunning.Count)
        Write-Output "START $taskId PID $($taskProcess.Id)"
    } catch {
        $taskRecord.status='launch_failed';$taskRecord.error=$_.Exception.Message
        $taskRecord.ended_utc=(Get-Date).ToUniversalTime().ToString('o')
    }
    Save-TaskManifest
}
Write-Output "MANIFEST $taskManifestPath"
while ($taskRunning.Count -gt 0) {
    foreach ($taskItem in @($taskRunning.ToArray())) {
        $taskItem.process.Refresh()
        if (-not $taskItem.process.HasExited) { continue }
        $taskItem.process.WaitForExit()
        $taskItem.record.exit_code=$taskItem.process.ExitCode
        $taskItem.record.ended_utc=(Get-Date).ToUniversalTime().ToString('o')
        $taskItem.record.status=if ($taskItem.process.ExitCode -eq 0) { 'process_completed' } else { 'process_failed' }
        $taskCompletion=Join-Path $taskOut "completed/$($taskItem.record.task_id).json"
        $taskItem.record.completion_exists=Test-Path -LiteralPath $taskCompletion
        if ($taskItem.record.completion_exists) {
            $taskItem.record.completion_sha256=(Get-FileHash -LiteralPath $taskCompletion -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        foreach ($taskStream in @('stdout','stderr')) {
            $taskStreamPath=Join-Path $taskRunDir $taskItem.record[$taskStream]
            $taskItem.record["${taskStream}_sha256"]=(Get-FileHash -LiteralPath $taskStreamPath -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        Write-Output "END $($taskItem.record.task_id) EXIT $($taskItem.record.exit_code)"
        $taskRunning.Remove($taskItem) | Out-Null
        Save-TaskManifest
    }
    if ($taskRunning.Count -gt 0) { Start-Sleep -Seconds 10 }
}
Save-TaskManifest
Write-Output 'ALL AUXILIARY PROCESSES FINISHED'
