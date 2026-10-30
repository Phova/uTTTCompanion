"""Long-span repetition diagnostics must respect document boundaries."""
import unittest

from deploy.corpus_repetition import analyze_repeated_spans, detected_language


class CorpusRepetitionTests(unittest.TestCase):
    def test_first_two_exact_spans_are_preserved_and_later_targets_are_counted(self):
        pattern = list(range(10, 26))
        result = analyze_repeated_spans([1, *pattern * 4, 2])
        self.assertEqual(result["mask"], [False] * 48 + [True] * 17 + [False])
        self.assertEqual(result["repeated_targets"], 17)
        self.assertEqual(result["supervised_targets"], 65)
        self.assertEqual(result["most_repeated_span_occurrences"], 4)
        self.assertEqual(result["longest_masked_run"], 17)

    def test_eos_bos_and_role_boundaries_reset_span_counts(self):
        pattern = list(range(10, 26))
        for boundary in (0, 1, 2, 3, 4, 5):
            with self.subTest(boundary=boundary):
                result = analyze_repeated_spans([1, *pattern * 2, boundary, *pattern * 2, 2])
                self.assertEqual(result["repeated_targets"], 0)
                self.assertFalse(any(result["mask"]))

    def test_nonrepeating_and_common_short_repetitions_are_not_flagged(self):
        self.assertEqual(analyze_repeated_spans([1, *range(10, 80), 2])["repeated_targets"], 0)
        self.assertEqual(analyze_repeated_spans([1, 10, 11, 10, 12, 10, 13, 10, 14, 2])["repeated_targets"], 0)

    def test_long_single_token_run_is_detected_but_special_tokens_are_protected(self):
        result = analyze_repeated_spans([1, *([10] * 24), 2, *([0] * 40)])
        self.assertEqual(result["repeated_targets"], 7)
        self.assertEqual(result["mask"][-41:], [False] * 41)

    def test_empty_windows_and_invalid_parameters(self):
        self.assertEqual(analyze_repeated_spans([])["fraction"], 0)
        for size, keep in ((1, 2), (16, 0)):
            with self.assertRaises(ValueError):
                analyze_repeated_spans([6, 7], size, keep)

    def test_detected_language_is_script_based(self):
        self.assertEqual(detected_language("科学研究的重要意义"), "zh_or_mixed_cjk")
        self.assertEqual(detected_language("The purpose of scientific research"), "latin_dominant")
        self.assertEqual(detected_language("12345 🙂"), "other_or_mixed")


if __name__ == "__main__":
    unittest.main()
