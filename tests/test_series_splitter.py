import unittest
import os
import shutil
from brain.memory import MovieState
from brain import config as cfg
from agents.video_merger_agent import find_smart_cut_points, _create_part_badge_png
from web_ui import StartRequest, BatchStartRequest


class TestSeriesSplitter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = os.path.abspath("temp_test_series")
        os.makedirs(self.temp_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_short_video_no_split(self):
        """A video shorter than max_duration (e.g. 120s <= 240s) should not be split."""
        cuts = find_smart_cut_points(
            total_duration=120.0,
            script_blocks=[],
            target_duration=180.0,
            min_duration=90.0,
            max_duration=240.0
        )
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0], (0.0, 120.0))

    def test_smart_cut_points_align_with_speech_gaps(self):
        """Smart cut points should choose gaps between speech blocks rather than cutting inside speech."""
        script_blocks = [
            {"start_sec": 0.0, "end_sec": 50.0},
            {"start_sec": 55.0, "end_sec": 175.0},
            {"start_sec": 185.0, "end_sec": 350.0},
            {"start_sec": 362.0, "end_sec": 540.0},
        ]

        cuts = find_smart_cut_points(
            total_duration=540.0,
            script_blocks=script_blocks,
            target_duration=180.0,
            min_duration=90.0,
            max_duration=240.0
        )

        self.assertEqual(len(cuts), 3)
        self.assertEqual(cuts[0][0], 0.0)
        self.assertAlmostEqual(cuts[0][1], 180.0, places=1)
        self.assertAlmostEqual(cuts[1][0], 180.0, places=1)
        self.assertAlmostEqual(cuts[1][1], 356.0, places=1)
        self.assertAlmostEqual(cuts[2][0], 356.0, places=1)
        self.assertEqual(cuts[2][1], 540.0)

        for i in range(len(cuts) - 1):
            self.assertEqual(cuts[i][1], cuts[i + 1][0])

    def test_small_remainder_merged_safely(self):
        """If the remaining tail is less than min_duration, it should merge into the prior episode."""
        cuts = find_smart_cut_points(
            total_duration=220.0,
            script_blocks=[{"start_sec": 0.0, "end_sec": 180.0}, {"start_sec": 185.0, "end_sec": 215.0}],
            target_duration=180.0,
            min_duration=90.0,
            max_duration=240.0
        )
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0], (0.0, 220.0))

    def test_create_part_badge_png(self):
        """Verify PNG badge generation with transparency and text."""
        badge_path = os.path.join(self.temp_dir, "test_badge.png")
        out = _create_part_badge_png("အပိုင်း ၁", is_portrait=False, output_path=badge_path)
        self.assertTrue(os.path.exists(out))
        self.assertGreater(os.path.getsize(out), 500)

        badge_en = os.path.join(self.temp_dir, "test_badge_en.png")
        out_en = _create_part_badge_png("Part 2", is_portrait=True, output_path=badge_en)
        self.assertTrue(os.path.exists(out_en))
        self.assertGreater(os.path.getsize(out_en), 500)

    def test_movie_state_series_fields(self):
        """Verify MovieState series serialization."""
        state = MovieState(movie_name="test_series_movie")
        state.series_enabled = True
        state.series_episodes = [
            {
                "part": 1,
                "part_label": "အပိုင်း ၁",
                "duration_sec": 180.0,
                "video_path": "outputs/test/series/part_01.mp4"
            }
        ]
        s_json = state.model_dump_json()
        restored = MovieState.model_validate_json(s_json)
        self.assertTrue(restored.series_enabled)
        self.assertEqual(len(restored.series_episodes), 1)
        self.assertEqual(restored.series_episodes[0]["part_label"], "အပိုင်း ၁")

    def test_web_request_models_series_support(self):
        """Verify StartRequest and BatchStartRequest accept series parameters."""
        req = StartRequest(
            input="movies/dummy.mp4",
            series_mode=True,
            series_duration=90
        )
        self.assertTrue(req.series_mode)
        self.assertEqual(req.series_duration, 90)

        batch_req = BatchStartRequest(
            inputs=["movies/dummy1.mp4", "movies/dummy2.mp4"],
            series_mode=True,
            series_duration=180
        )
        self.assertTrue(batch_req.series_mode)
        self.assertEqual(batch_req.series_duration, 180)

    def test_has_audio_stream_check(self):
        """Verify _has_audio_stream handles nonexistent or valid files properly."""
        from agents.video_merger_agent import _has_audio_stream
        self.assertFalse(_has_audio_stream("non_existent_file.mp4"))
        self.assertFalse(_has_audio_stream(""))

    def test_thumbnail_agent_fallback_on_invalid_file(self):
        """Verify generate_episode_thumbnail returns None safely when video cannot be opened."""
        from agents.thumbnail_agent import ThumbnailAgent
        agent = ThumbnailAgent()
        state = MovieState(movie_name="test_thumb_movie")
        res = agent.generate_episode_thumbnail(
            state=state,
            movie_path="non_existent_video.mp4",
            part_num=1
        )
        self.assertIsNone(res)


if __name__ == "__main__":
    unittest.main()

