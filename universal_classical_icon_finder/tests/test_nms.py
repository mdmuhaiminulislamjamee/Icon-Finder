from detector.candidate import Candidate
from detector.nms import iou, non_max_suppression
from detector.candidate_fusion import fuse_candidates


def candidate(name="play", box=(10, 10, 30, 30), confidence=0.9, method="template_matching"):
    return Candidate(name, "generic", method, confidence, confidence, box, 1, "play.png")


def test_iou_and_empty_inputs():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0
    assert non_max_suppression([]) == []


def test_fusion_preserves_distinct_instances_and_identities():
    strongest = candidate(confidence=0.99)
    duplicates = [strongest, candidate(box=(11, 11, 31, 31), method="edge_matching"),
                  candidate(box=(100, 100, 120, 120)), candidate("pause")]
    fused = non_max_suppression(fuse_candidates(duplicates))
    assert len(fused) == 3
    assert strongest.confidence == 0.99
    assert strongest.supporting_methods == ["edge_matching", "template_matching"]
