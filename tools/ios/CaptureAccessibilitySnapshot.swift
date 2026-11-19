// Add this file to the original MediaCollector XCTest target on a Mac.
// It is an integration helper, not a replacement or a compiled app update.
import CryptoKit
import Foundation
import UIKit
import XCTest

enum CaptureAccessibilitySnapshot {
    enum SnapshotError: Error { case invalidCapture, invalidScreenFrame }

    /// Call while the target app is still foregrounded, immediately after
    /// taking the screenshot. Pass the EXACT PNG bytes written to images/.
    /// screenFrame is the full screenshot's rectangle in XCTest screen points,
    /// NOT the player rectangle or the app window in Split View.
    @MainActor
    static func write(
        app: XCUIApplication,
        pngData: Data,
        screenFrame: CGRect,
        recordID: String,
        collectedDirectory: URL,
        screenshotCapturedAt: Date? = nil
    ) throws -> URL {
        guard recordID.range(of: #"^[A-Za-z0-9][A-Za-z0-9_.-]*$"#, options: .regularExpression) != nil,
              let image = UIImage(data: pngData)?.cgImage else {
            throw SnapshotError.invalidCapture
        }
        guard screenFrame.width > 0, screenFrame.height > 0 else {
            throw SnapshotError.invalidScreenFrame
        }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let started = formatter.string(from: Date())
        // Preserve the complete XCTest tree text as well as structured fields.
        // Querying elements is sequential: this is not an atomic video snapshot.
        let debugTree = app.debugDescription
        let candidates: [XCUIElement] = [app] + app.descendants(matching: .any).allElementsBoundByIndex
        var elements: [[String: Any]] = []
        for (index, element) in candidates.enumerated() {
            guard element.exists else { continue }
            let frame = element.frame
            guard [frame.minX, frame.minY, frame.width, frame.height].allSatisfy({ $0.isFinite }) else { continue }
            let hittable = element.isHittable
            elements.append([
                "element_index": index,
                "identifier": element.identifier,
                "label": element.label,
                "value": element.value.map { String(describing: $0) } ?? "",
                "element_type": element.elementType.rawValue,
                "element_type_name": String(describing: element.elementType),
                "frame": [frame.minX, frame.minY, frame.width, frame.height],
                // This is the interaction frame, NOT a localized icon box.
                // Optional visual_xyxy must be independently localized and
                // expressed in original screenshot pixels, never screen points.
                "visible": hittable,
                "is_hittable": hittable,
                "enabled": element.isEnabled
            ])
        }
        var payload: [String: Any] = [
            "schema": "media_collector_accessibility_v1",
            "record_id": recordID,
            "image_sha256": SHA256.hash(data: pngData).map { String(format: "%02x", $0) }.joined(),
            "image_size": [image.width, image.height],
            "coordinate_space": "screen_points",
            "screen_frame": [screenFrame.minX, screenFrame.minY, screenFrame.width, screenFrame.height],
            "snapshot_started_at": started,
            "snapshot_finished_at": formatter.string(from: Date()),
            "source": "ios_xctest_accessibility_full_snapshot",
            "geometry_policy_version": "media_control_visual_geometry_v1",
            "visibility_policy": "isHittable is a conservative proxy; non-hittable elements are retained for review",
            "elements": elements,
            "debug_tree": debugTree
        ]
        if let screenshotCapturedAt {
            payload["screenshot_captured_at"] = formatter.string(from: screenshotCapturedAt)
        }
        let directory = collectedDirectory.appendingPathComponent("accessibility", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let destination = directory.appendingPathComponent(recordID + ".json")
        let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: destination, options: .atomic)
        return destination
    }
}
