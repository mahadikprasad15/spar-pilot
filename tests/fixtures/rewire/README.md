# Rewire module shape fixture

`config.json` is the model config of the pinned base, `Qwen/Qwen2.5-1.5B-Instruct` at revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, transcribed for offline tests (the Hub is not reachable from the
CI container). The rewire tests read MLP shapes only from this file, never from a download.

`tests/test_rewire_module.py::test_shape_fixture_matches_the_pinned_config_when_present` compares the shape
fields with the real `config.json` under `$PILOT_ARTIFACTS/inputs/models/989aa79…/` whenever that directory
exists, so a transcription error is caught on any machine that has the pinned files.

It sets no `pad_token_id`, like the real file (lesson 2026-11-11: fixtures must not supply what the model lacks).
