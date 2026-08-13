import unittest

from src.services.skill_canonicalization import SkillCanonicalizer


class SkillCanonicalizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canonicalizer = SkillCanonicalizer()

    def test_normalizes_case_separators_and_spaces(self) -> None:
        self.assertEqual(
            self.canonicalizer.canonicalize("  POSTGRE_SQL  "),
            "postgresql",
        )

    def test_resolves_english_and_russian_aliases(self) -> None:
        examples = {
            "Golang": "go",
            "Голанг": "go",
            "Постгрес": "postgresql",
            "Кубер": "kubernetes",
            "Реакт Джс": "react",
            "Яндекс Облако": "yandex cloud",
            "Машинное обучение": "machine learning",
            "Компьютерное зрение": "computer vision",
        }

        for alias, expected in examples.items():
            with self.subTest(alias=alias):
                self.assertEqual(self.canonicalizer.canonicalize(alias), expected)

    def test_preserves_meaningful_punctuation(self) -> None:
        for skill in ("C++", "C#", ".NET", "Node.js"):
            with self.subTest(skill=skill):
                expected = skill.casefold()
                self.assertEqual(self.canonicalizer.canonicalize(skill), expected)

    def test_does_not_replace_alias_inside_a_phrase(self) -> None:
        self.assertEqual(
            self.canonicalizer.canonicalize("Опыт работы с Postgres"),
            "опыт работы с postgres",
        )

    def test_normalizes_custom_aliases(self) -> None:
        canonicalizer = SkillCanonicalizer({"Vue_JS": "Vue.js"})

        self.assertEqual(canonicalizer.canonicalize("vue-js"), "vue.js")

    def test_many_deduplicates_aliases_and_ignores_empty_values(self) -> None:
        self.assertEqual(
            self.canonicalizer.canonicalize_many(
                ["Postgres", "POSTGRE_SQL", "Постгрес", "  "],
            ),
            {"postgresql"},
        )


if __name__ == "__main__":
    unittest.main()
