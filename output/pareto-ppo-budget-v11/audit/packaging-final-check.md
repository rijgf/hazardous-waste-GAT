# Final Git-index export verification

After publishing the formal Markdown, exported the actual staged Git index into a new isolated temporary directory, rather than testing only the existing dirty workspace.

- Export directory: `C:/Users/YANGFE~1/AppData/Local/Temp/ppo-v11-final-index-028755bad9e045ac8906f81f72bf7312`.
- `py -B -m unittest discover -s tests -q`: **171 tests PASS**, 13.547 seconds, exit 0.
- `run_ppo_budget_v11.load_protocol()` from the exported directory: **PASS** for frozen computation source, both V9 model files, instances and protected reused MILP file hashes.
- No training or formal optimization experiment was rerun in this check. Tests include small synthetic verification fixtures only.
- An initial PowerShell argument-tokenization error prevented the first checkout-index invocation from exporting; the argument was corrected and the same newly created empty directory was successfully exported. This did not modify any experiment.
- The earlier clean-index precheck exposed one missing historical V8 registry dependency; it was explicitly included before this successful final export. It does not replace the actual V9 checkpoints.

The final numeric-with-report and independent delivery certificates finish after this export; they are added to the same submission afterward. Those late audit records do not change the already tested computation/report source or frozen data. Final staged-file checks must still confirm the published manuscript and both model hashes, no newly edited Word/PDF artifacts, and no file exceeding GitHub's individual-file limit.
