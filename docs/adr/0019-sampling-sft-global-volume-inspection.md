# 0019: Verify Global Volume storage through Runpod REST v2

Date: 2026-10-31

The user attached a Global Volume to the A40 Pod. runpodctl 3.0.1's Pod output
omits that attachment (networkVolumeId is null). When the legacy CLI does not
report a network volume, inspect GET https://api.runpod.io/v2/pods/{id} using the
existing RUNPOD_API_KEY. Require the same Pod ID and exactly one network/global
mount with an explicit absolute path and volume ID. Verify the artifact root is
inside that mount. Never assume missing storage means durable storage exists.

Keep the saved request's legacy network_volume_id field for compatibility; it now
identifies the selected persistent volume, including global volumes. The provider
returns explicit storageType, storageVolumeId and storageMountPath. Preserve the
legacy networkVolumeId field without pretending a global volume is regional.
Do not export the API response's environment or bearer token. Pod stop still uses
the existing guarded CLI; no provisioning or deletion authority is added.

Source: https://docs.runpod.io/api-reference-v2/pods/get-a-pod
