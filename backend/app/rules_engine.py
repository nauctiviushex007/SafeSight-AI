"""
Rules engine: turns raw detections into *potential* violations.
Pure Python (no FastAPI/DB) so it is easy to unit-test and explain.

Limitations (be honest about these in demos):
- Helmet/vest-to-person matching is geometric and can be wrong in crowds.
- Occlusion, poor light, and tiny people reduce reliability -> we flag "uncertain".
- Zone checks use the bottom-centre of the person box as an approximate foot
  position; this ignores camera perspective unless zones are drawn per camera.
"""
from dataclasses import dataclass
from . import config


@dataclass
class Detection:
    label: str                      # "person" | "helmet" | "vest"
    confidence: float
    bbox: tuple[float, float, float, float]

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]


@dataclass
class Violation:
    category: str
    severity: str
    confidence: float
    certainty: str                  # "likely" | "uncertain"
    description: str
    person_bbox: tuple


def point_in_polygon(point, polygon) -> bool:
    """Ray-casting algorithm."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _helmet_matches(person: Detection, helmet: Detection) -> bool:
    """Helmet centre must be inside the person box, in its upper 35%."""
    x1, y1, x2, y2 = person.bbox
    cx, cy = helmet.center
    return x1 <= cx <= x2 and y1 <= cy <= y1 + 0.35 * (y2 - y1)


def _vest_matches(person: Detection, vest: Detection) -> bool:
    """Vest centre must be inside the person box, in the torso region (15%-70%)."""
    x1, y1, x2, y2 = person.bbox
    cx, cy = vest.center
    h = y2 - y1
    return x1 <= cx <= x2 and y1 + 0.15 * h <= cy <= y1 + 0.70 * h


def evaluate(detections, restricted_zones=None, supports=("helmet", "vest")):
    """
    detections: list[Detection]
    supports: PPE classes the loaded model can actually detect. We never report
              a missing-PPE violation for a class the model cannot detect.
    """
    zones = restricted_zones or []
    people = [d for d in detections if d.label == "person"]
    helmets = [d for d in detections if d.label == "helmet" and d.confidence >= config.MIN_PPE_CONF]
    vests = [d for d in detections if d.label == "vest" and d.confidence >= config.MIN_PPE_CONF]
    out: list[Violation] = []

    for p in people:
        if p.confidence < config.MIN_PERSON_CONF:
            continue
        # Small or low-confidence people -> evidence is weak -> human review
        weak = p.height < config.MIN_PERSON_HEIGHT_PX or p.confidence < 0.65
        certainty = "uncertain" if weak else "likely"
        note = " (small/low-confidence detection; verify manually)" if weak else ""

        if "helmet" in supports and not any(_helmet_matches(p, h) for h in helmets):
            out.append(Violation("no_helmet", config.SEVERITY_RULES["no_helmet"], p.confidence,
                                 certainty, "Worker detected without an associated helmet" + note, p.bbox))
        if "vest" in supports and not any(_vest_matches(p, v) for v in vests):
            out.append(Violation("no_vest", config.SEVERITY_RULES["no_vest"], p.confidence,
                                 certainty, "Worker detected without an associated safety vest" + note, p.bbox))

        x1, y1, x2, y2 = p.bbox
        feet = ((x1 + x2) / 2, y2)
        for zone in zones:
            if point_in_polygon(feet, zone):
                out.append(Violation("restricted_zone", config.SEVERITY_RULES["restricted_zone"],
                                     p.confidence, certainty,
                                     "Person's estimated position is inside a restricted zone" + note, p.bbox))
                break
    return out
