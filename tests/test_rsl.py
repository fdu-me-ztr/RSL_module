import unittest
from rsl import solve, retrieve, parse_review
import json


class FlowTests(unittest.TestCase):
    def fake(self, values):
        iterator = iter(values)
        return lambda *args: next(iterator)

    def review(self, passed):
        return json.dumps(dict(score=9 if passed else 4, passed=passed,
                               issues=[] if passed else ['计算错误'], lesson='检查算术'))

    def test_recursive_revision(self):
        r = solve('计算', self.fake(['错误', self.review(False), '正确', self.review(True)]))
        self.assertEqual(r['answer'], '正确')
        self.assertEqual(len(r['history']), 2)
        self.assertEqual(r['status'], 'self_review_passed')

    def test_limit(self):
        r = solve('计算', self.fake(['错误', self.review(False)]), max_rounds=1)
        self.assertEqual(r['status'], 'max_rounds')

    def test_invalid_review_fails_closed(self):
        r = solve('计算', self.fake(['答案', '{}']))
        self.assertEqual(r['status'], 'invalid_review')

    def test_no_change(self):
        r = solve('计算', self.fake(['答案', self.review(False), '答案']))
        self.assertEqual(r['status'], 'no_change')

    def test_memory_matching(self):
        records = [dict(question='Verilog同步复位', lesson='观察时钟边沿')]
        self.assertEqual(len(retrieve('Verilog复位', records)), 1)
        self.assertEqual(retrieve('apple banana', records), [])

    def test_bad_scores(self):
        with self.assertRaises(ValueError):
            parse_review('{"score": true}')


if __name__ == '__main__':
    unittest.main()
