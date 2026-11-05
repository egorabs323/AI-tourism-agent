import unittest
from evaluate_rag import score_rows

class RetrievalMetricTests(unittest.TestCase):
    def test_precision_and_pooled_recall(self):
        rows = [dict(query_id='R01', rank=str(i), relevant=str(v)) for i,v in enumerate([1,0,1,0,1,1],1)]
        score=score_rows(rows,5)[0]
        self.assertEqual(score['precision_at_k'],0.6)
        self.assertEqual(score['pooled_recall_at_k'],0.75)

    def test_no_labels_is_error(self):
        with self.assertRaisesRegex(ValueError,'Не размечен'):
            score_rows([dict(query_id='R01',rank='1',relevant='')],1)

    def test_zero_relevant_no_recall(self):
        rows=[dict(query_id='T',rank=str(i),relevant='0') for i in range(1,6)]
        result=score_rows(rows,5)[0]
        self.assertEqual(result['precision_at_k'],0)
        self.assertEqual(result['pooled_recall_at_k'],'')

if __name__=='__main__':
    unittest.main()
