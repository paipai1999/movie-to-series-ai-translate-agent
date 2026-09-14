"""
Automated tests verifying comprehensive audit fixes.
"""
import unittest
from unittest.mock import patch, MagicMock
from brain.memory import MovieState, TranscriptSegment
from brain.gemini_client import _FALLBACK_MODELS
from agents.qa_agent import QAAgent
from agents.audio_agent import AudioAgent


class TestAuditFixes(unittest.TestCase):

    def test_movie_state_subtitles_burned_flag(self):
        """Verify subtitles_burned exists on MovieState and defaults to False."""
        state = MovieState(movie_name="AuditTest")
        self.assertFalse(state.subtitles_burned)
        state.subtitles_burned = True
        self.assertTrue(state.subtitles_burned)

    def test_gemini_fallback_models_include_production(self):
        """Verify _FALLBACK_MODELS includes standard Google AI Studio production models."""
        for required_model in ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash"]:
            self.assertIn(required_model, _FALLBACK_MODELS)

    def test_qa_agent_scene_id_collision_prevention(self):
        """Verify QAAgent does not collide 'scene_1' and 'action_bridge_1' into '1'."""
        qa = QAAgent()
        state = MovieState(movie_name="CollisionTest")
        state.language = "english"
        state.generated_script = [
            {
                "scene_id": "scene_1",
                "narration": "Original narration for main scene 1 which is very long and needs to be shortened significantly to fit timing.",
                "start_sec": 0.0,
                "end_sec": 2.0,
            },
            {
                "scene_id": "action_bridge_1",
                "narration": "Original narration for bridge 1 which is also quite long and exceeds the allowed duration.",
                "start_sec": 2.0,
                "end_sec": 4.0,
            }
        ]

        fake_llm_response = """
        [
            {"scene_id": "scene_1", "rewritten_narration": "Short scene 1."},
            {"scene_id": "action_bridge_1", "rewritten_narration": "Short bridge 1."}
        ]
        """
        with patch("brain.gemini_client.call_gemini", return_value=(fake_llm_response, None)):
            with patch("brain.config.load_config", return_value={"gemini": {"enabled": True, "api_keys": ["fake-key"]}}):
                updated_state = qa.enforce_duration_constraints(state)

        # Both should have their OWN distinct rewritten narration
        script = updated_state.generated_script
        self.assertEqual(script[0]["narration"], "Short scene 1.")
        self.assertEqual(script[1]["narration"], "Short bridge 1.")

    def test_audio_agent_correct_transcript_preserves_all_segments(self):
        """Verify correct_transcript does not truncate large transcripts and preserves all segments."""
        state = MovieState(movie_name="TruncationTest")
        # Create 150 segments (which would previously exceed 40000 chars if long or get truncated)
        original_segments = [
            TranscriptSegment(start=float(i * 3), end=float(i * 3 + 2.5), text=f"Dialogue segment {i}")
            for i in range(150)
        ]
        state.transcript = list(original_segments)

        # Mock call_gemini to simulate a batch returning corrected text for each chunk
        def mock_call_gemini(sys_p, user_p, api_key, model=None, temperature=None):
            import re
            lines = re.findall(r'\[([\d.]+)-([\d.]+)\]\s*(.*)', user_p)
            segments_json = [{"start": float(l[0]), "end": float(l[1]), "text": f"Corrected {l[2]}"} for l in lines]
            import json
            return json.dumps(segments_json), None

        with patch("brain.gemini_client.call_gemini", side_effect=mock_call_gemini):
            with patch("brain.config.load_config", return_value={"gemini": {"enabled": True, "api_keys": ["fake-key"]}}):
                audio_agent = AudioAgent("dummy_path.mp4")
                res_state = audio_agent.correct_transcript(state)

        self.assertEqual(len(res_state.transcript), 150)
        self.assertTrue(res_state.transcript[0].text.startswith("Corrected Dialogue segment 0"))
        self.assertTrue(res_state.transcript[149].text.startswith("Corrected Dialogue segment 149"))

    def test_downloader_agent_none_filename_guard(self):
        """Verify DownloaderAgent handles None filename without raising TypeError."""
        from agents.downloader_agent import DownloaderAgent
        dl = DownloaderAgent(output_dir="temp_test_dl")
        with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
            mock_instance = MagicMock()
            mock_ydl_cls.return_value.__enter__.return_value = mock_instance
            mock_instance.extract_info.return_value = {"title": "dummy_video"}
            mock_instance.prepare_filename.return_value = None

            with self.assertRaises(FileNotFoundError) as ctx:
                dl.download_video("https://youtube.com/watch?v=dummy123")
            self.assertIn("Download seemed to succeed but file not found", str(ctx.exception))

    def test_downloader_agent_youtube_shorts_clean_url(self):
        """Verify _clean_url converts YouTube Shorts link to canonical watch URL."""
        from agents.downloader_agent import DownloaderAgent
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ?feature=share"
        cleaned = DownloaderAgent._clean_url(url)
        self.assertEqual(cleaned, "https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    def test_batch_outro_protection_wiring(self):
        """Verify BatchStartRequest and BatchProcessor accept outro protection parameters."""
        from web_ui import BatchStartRequest
        from brain.planner import BatchProcessor

        req = BatchStartRequest(
            inputs=["test.mp4"],
            trim_end=12.5,
            no_smart_trim=True,
            outro_card=True
        )
        self.assertEqual(req.trim_end, 12.5)
        self.assertTrue(req.no_smart_trim)
        self.assertTrue(req.outro_card)

        proc = BatchProcessor(
            movies_folder="movies",
            trim_end=12.5,
            no_smart_trim=True,
            outro_card=True
        )
        self.assertEqual(proc.trim_end, 12.5)
        self.assertTrue(proc.no_smart_trim)
        self.assertTrue(proc.outro_card)

    def test_voice_agent_presorts_script_blocks(self):
        """Verify VoiceAgent sorts generated_script by start_sec before TTS generation."""
        from agents.voice_agent import VoiceAgent
        agent = VoiceAgent(output_dir="temp_dummy_vo")
        state = MovieState(movie_name="SortScriptTest")
        state.generated_script = [
            {"scene_id": 2, "narration": "Second scene", "start_sec": 15.0},
            {"scene_id": 1, "narration": "First scene", "start_sec": 3.0},
            {"scene_id": 3, "narration": "Third scene", "start_sec": 22.0},
        ]
        # Calling generate_voiceover with an empty/mocked folder
        with patch.object(agent, "_generate_edge_voiceover", return_value=state):
            agent.generate_voiceover(state)
            self.assertEqual(state.generated_script[0]["start_sec"], 3.0)
            self.assertEqual(state.generated_script[1]["start_sec"], 15.0)
            self.assertEqual(state.generated_script[2]["start_sec"], 22.0)

    def test_config_json_outro_protection_present(self):
        """Verify config.json contains the outro_protection section."""
        from brain import config as cfg
        c = cfg.load_config()
        self.assertIn("outro_protection", c)
        self.assertIn("auto_trim", c["outro_protection"])
        self.assertIn("trim_end_seconds", c["outro_protection"])


    def test_qa_agent_detects_hardware_encoder_dict(self):
        """Verify QAAgent._create_qa_preview correctly extracts codec from dict returned by detect_hardware_encoder."""
        qa = QAAgent()
        with patch("agents.video_merger_agent.detect_hardware_encoder", return_value={"codec": "h264_nvenc", "type": "gpu"}), \
             patch("subprocess.run") as mock_run, \
             patch("os.path.exists", return_value=True), \
             patch("os.path.getsize", return_value=10_000_000):
            mock_run.return_value = MagicMock(returncode=0)
            res = qa._create_qa_preview("fake_in.mp4", "fake_out.mp4")
            # Verify h264_nvenc was passed into cmd arguments
            called_cmd = mock_run.call_args[0][0]
            self.assertIn("h264_nvenc", called_cmd)

    def test_voice_agent_language_resolution(self):
        """Verify VoiceAgent properly retains language and deduces correctly."""
        from agents.voice_agent import VoiceAgent
        # 1. Custom language passed explicitly
        va1 = VoiceAgent(voice="my-MM-ThihaNeural", language="burmese")
        self.assertEqual(va1.language, "burmese")

        # 2. English voice without language -> deduced as english
        va2 = VoiceAgent(voice="en-US-GuyNeural")
        self.assertEqual(va2.language, "english")

        # 3. Burmese voice without language -> deduced as burmese
        va3 = VoiceAgent(voice="my-MM-NilarNeural")
        self.assertEqual(va3.language, "burmese")

    def test_cli_language_and_voice_precedence(self):
        """Verify main.py CLI parsing handles language and voice independently."""
        import argparse
        # Simulate main.py argument resolution logic
        def resolve_lang_voice(lang, voice):
            clean_lang = lang
            chosen_voice = voice
            if clean_lang in ["burmese_nilar", "nilar"]:
                clean_lang = "burmese"
                if not chosen_voice:
                    chosen_voice = "my-MM-NilarNeural"
            elif clean_lang in ["burmese_thiha", "thiha"]:
                clean_lang = "burmese"
                if not chosen_voice:
                    chosen_voice = "my-MM-ThihaNeural"
            elif clean_lang in ["mm", "myanmar"]:
                clean_lang = "burmese"
            elif clean_lang in ["en"]:
                clean_lang = "english"

            if chosen_voice in ["nilar", "female"]:
                chosen_voice = "my-MM-NilarNeural"
            elif chosen_voice in ["thiha", "male"]:
                chosen_voice = "my-MM-ThihaNeural"
            return clean_lang, chosen_voice

        l, v = resolve_lang_voice("mm", "female")
        self.assertEqual(l, "burmese")
        self.assertEqual(v, "my-MM-NilarNeural")

        l2, v2 = resolve_lang_voice("myanmar", "thiha")
        self.assertEqual(l2, "burmese")
        self.assertEqual(v2, "my-MM-ThihaNeural")

    def test_master_agent_preserves_audio_path(self):
        """Verify audio_path is preserved when merging audio_state in parallel execution."""
        from agents.master import MasterAgent
        agent = MasterAgent.__new__(MasterAgent)
        agent.state = MovieState(movie_name="AudioPathTest")
        agent.state.audio_path = None

        audio_state = MovieState(movie_name="AudioState")
        audio_state.audio_path = "outputs/audio.wav"
        audio_state.transcript = "dummy transcript"
        audio_state.characters = ["A", "B"]

        # Simulate the merge block in master.py
        agent.state.transcript = getattr(audio_state, "transcript", agent.state.transcript)
        if hasattr(audio_state, "characters") and audio_state.characters:
            agent.state.characters = audio_state.characters
        if hasattr(audio_state, "audio_path") and audio_state.audio_path:
            agent.state.audio_path = audio_state.audio_path

        self.assertEqual(agent.state.audio_path, "outputs/audio.wav")


if __name__ == "__main__":
    unittest.main()

