"""Regression tests for loopback-only Fast DDS & PX4 XRCE isolation configuration."""
from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BRINGUP_DIR = _REPO_ROOT / "src" / "swarm_bringup"
_PROFILE_PATH = _BRINGUP_DIR / "config" / "fastdds_loopback.xml"
_LAUNCH_PATH = _BRINGUP_DIR / "launch" / "swarm.launch.py"
_SETUP_PATH = _BRINGUP_DIR / "setup.py"


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_child(elem: ET.Element, name: str) -> ET.Element | None:
    for child in elem:
        if _strip_ns(child.tag) == name:
            return child
    return None


def _findall_children(elem: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in elem if _strip_ns(child.tag) == name]


class TestDdsLoopbackIsolationConfig(unittest.TestCase):
    def test_fastdds_loopback_xml_schema_and_whitelist(self) -> None:
        self.assertTrue(_PROFILE_PATH.is_file(), f"Missing Fast DDS profile: {_PROFILE_PATH}")
        # MicroXRCEAgent ArgumentParser rejects '-r' arguments containing hyphens
        self.assertNotIn("-", _PROFILE_PATH.name)

        root = ET.parse(_PROFILE_PATH).getroot()
        self.assertEqual(_strip_ns(root.tag), "dds")

        profiles = _find_child(root, "profiles")
        self.assertIsNotNone(profiles, "<profiles> must be a child of <dds>")
        assert profiles is not None

        transports = _find_child(profiles, "transport_descriptors")
        self.assertIsNotNone(transports, "<transport_descriptors> must be inside <profiles>")
        assert transports is not None

        descriptors = _findall_children(transports, "transport_descriptor")
        self.assertEqual(len(descriptors), 1)
        desc = descriptors[0]

        transport_id = _find_child(desc, "transport_id")
        transport_type = _find_child(desc, "type")
        whitelist = _find_child(desc, "interfaceWhiteList")
        self.assertIsNotNone(transport_id)
        self.assertIsNotNone(transport_type)
        self.assertIsNotNone(whitelist)
        assert transport_id is not None and transport_type is not None and whitelist is not None

        self.assertEqual((transport_id.text or "").strip(), "udp_localhost")
        self.assertEqual((transport_type.text or "").strip(), "UDPv4")

        addresses = [
            (addr.text or "").strip() for addr in _findall_children(whitelist, "address")
        ]
        self.assertEqual(
            addresses,
            ["127.0.0.1"],
            "interfaceWhiteList must strictly allow only loopback 127.0.0.1",
        )

        participants = _findall_children(profiles, "participant")
        px4_participants = [
            p for p in participants if p.attrib.get("profile_name") == "px4_participant"
        ]
        self.assertEqual(
            len(px4_participants),
            1,
            "Expected exactly one participant profile named 'px4_participant'",
        )
        participant = px4_participants[0]
        self.assertEqual(participant.attrib.get("is_default_profile"), "true")

        rtps = _find_child(participant, "rtps")
        self.assertIsNotNone(rtps)
        assert rtps is not None

        use_builtin = _find_child(rtps, "useBuiltinTransports")
        user_transports = _find_child(rtps, "userTransports")
        self.assertIsNotNone(use_builtin)
        self.assertIsNotNone(user_transports)
        assert use_builtin is not None and user_transports is not None

        self.assertEqual((use_builtin.text or "").strip().lower(), "false")
        ref_ids = [
            (tid.text or "").strip()
            for tid in _findall_children(user_transports, "transport_id")
        ]
        self.assertEqual(ref_ids, ["udp_localhost"])

    def test_launch_and_setup_wire_loopback_profile_and_px4_ptcfg(self) -> None:
        setup_text = _SETUP_PATH.read_text(encoding="utf-8")
        self.assertIn("config/fastdds_loopback.xml", setup_text)

        launch_text = _LAUNCH_PATH.read_text(encoding="utf-8")
        self.assertIn("fastdds_loopback.xml", launch_text)
        self.assertIn("FASTDDS_DEFAULT_PROFILES_FILE", launch_text)
        self.assertIn('"PX4_PARAM_UXRCE_DDS_PTCFG": "2"', launch_text)
        self.assertIn('"PX4_PARAM_UXRCE_DDS_AG_IP": "2130706433"', launch_text)


if __name__ == "__main__":
    unittest.main()
