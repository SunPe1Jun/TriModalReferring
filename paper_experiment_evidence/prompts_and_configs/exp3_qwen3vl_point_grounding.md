You are performing candidate-free point-supervised 3D referent grounding for egocentric VR interaction.

Return strict JSON only. Your entire response must start with "{" and end with "}". Do not use markdown, code fences, comments, notes, or natural-language paragraphs outside the JSON object. Stop immediately after the final closing brace.

Task:
Given the user instruction, up to three chronological evidence frames, and measured camera/gaze/hand telemetry in the same Unity world coordinate system, predict the 3D world-coordinate point or points corresponding to the intended referent or referents.

Important constraints:
- You are not given candidate anchors.
- You are not given ground-truth coordinates or object names.
- Infer the number of intended referents from the instruction and evidence.
- Gaze and hand ray endpoints are behavioral pointing cues, not ground truth and not guaranteed object centers.
- Camera pose gives the viewing direction and scale context.
- Use the scene robust bounds only as a broad coordinate sanity range, not as object locations.
- If evidence is insufficient, return an empty points_3d array.

Coordinate system:
- Coordinates are Unity world coordinates.
- Unit is world unit unless the data audit explicitly verifies meters.
- +X, +Y, and +Z follow the released Unity telemetry and anchor-table convention.

Required output schema:
{
  "points_3d": [
    {
      "referent": "short object or spatial referent description",
      "point": [0.0, 0.0, 0.0],
      "confidence": 0.5
    }
  ]
}

Schema rules:
- "points_3d" must be an array.
- Each point must contain exactly three finite numbers [x, y, z].
- confidence is optional, but if present it must be a finite number in [0, 1].
- Do not output direction vectors.
- Do not output 2D image points.
- Do not output anchor ids.
- Do not include long reasoning. The "referent" field should be short.

Point format hard constraints:
- Every "point" value must be exactly one coordinate triple [x_world, y_world, z_world].
- Never output 4, 6, 7, or 9 values in a point.
- Never concatenate multiple cues into one point.
- Never append direction vectors, camera basis vectors, quaternions, bbox values, distances, confidence, origin, or time values into point.
- Do not copy fields marked do_not_copy as output points.
- The only telemetry fields intended for direct copying are under `primary_copyable_gaze_point_hypotheses` and `secondary_copyable_hand_point_hypotheses`.
- Do not use camera_position_world or ray origins as targets unless the instruction explicitly refers to the user/camera/controller location.

Grounding procedure:
- This diagnostic evaluates target-free measured 3D point hypotheses, not free-form 3D reconstruction.
- Each non-null entry under `primary_copyable_gaze_point_hypotheses` has a hypothesis id such as P1_GAZE and a point array.
- Default behavior: output all distinct non-null primary gaze hypotheses in chronological order, up to three entries. Put the hypothesis id in the `referent` field and copy its point array exactly.
- Use `secondary_copyable_hand_point_hypotheses` only if there are no non-null gaze hypotheses. Put the hand hypothesis id in the `referent` field and copy its point array exactly.
- Do not synthesize object-center coordinates for this diagnostic. The model's job is to select/copy measured point hypotheses that are consistent with the instruction and evidence.
- Do not invent arbitrary canonical coordinates such as [0, -0.5, 15] or [1, -0.5, 15].
- If two hypotheses have identical or nearly identical points, output only the earliest one; otherwise keep distinct hypotheses separate.

Invalid point examples:
- {"points_3d":[{"referent":"object","point":[1,2,3,0,0,1]}]}
- {"points_3d":[{"referent":"object","point":[1,2,3,1,2,4]}]}
- {"points_3d":[{"referent":"object","point":[1,2,3,0.7]}]}

Valid point example:
- {"points_3d":[{"referent":"object","point":[1.0,2.0,3.0],"confidence":0.7}]}

Event:
[EVENT_BLOCK]

Scene bounds:
[SCENE_BOUNDS_BLOCK]

Evidence frames and telemetry:
[EVIDENCE_BLOCK]

Remember: return only JSON. Each point must contain exactly three numeric coordinates. Default to copying all distinct non-null P*_GAZE hypotheses exactly; do not generate free-form coordinates.
