import unittest
import tempfile
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from spaceoracle.oracles import OracleQueue

class TestOracleQueue(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.all_genes = ['gene1', 'gene2', 'gene3', 'gene4', 'gene5']
        self.queue = OracleQueue(self.temp_dir, self.all_genes)

    def tearDown(self):
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_extract_gene_name(self):
        test_cases = [
            ('/folder1/folder2/genez_estimator.pkl', 'genez'),
            ('/folder1/folder2/gene_extra__estimator.pkl', 'gene_extra_'),
            ('gene_4567_estimator.pkl', 'gene_4567'),
            ('gene_4567_estimator', None),
            ('gene_4567_other.lock', None)
        ]

        for input_path, expected_output in test_cases:
            with self.subTest(input_path=input_path):
                self.assertEqual(
                    OracleQueue.extract_gene_name(input_path),
                    expected_output
                )

        self.assertEqual(
            OracleQueue.extract_gene_name('/folder1/folder2/genez_estimator.pkl'),
            self.queue.extract_gene_name('/folder1/folder2/genez_estimator.pkl')
        )



    def test_initial_state(self):
        self.assertEqual(len(self.queue), 5)
        self.assertFalse(self.queue.is_empty)

    def test_next(self):
        gene = next(self.queue)
        self.assertIn(gene, self.all_genes)
        

    def _create_estimator_pkl(self, gene):
        with open(f'{self.temp_dir}/{gene}_estimator.pkl', 'w') as f:
            f.write('dummy data')

    def test_create_and_delete_lock(self):
        gene = 'gene1'
        self.queue.create_lock(gene)
        self.assertTrue(os.path.exists(f'{self.temp_dir}/{gene}.lock'))
        self.assertEqual(len(self.queue), len(self.all_genes))

        self._create_estimator_pkl(gene)
        self.assertTrue(os.path.exists(f'{self.temp_dir}/{gene}_estimator.pkl'))
        self.assertTrue(os.path.exists(f'{self.temp_dir}/{gene}.lock'))

        self.queue.delete_lock(gene)
        self.assertFalse(os.path.exists(f'{self.temp_dir}/{gene}.lock'))
        self.assertEqual(len(self.queue), len(self.all_genes)-1)

    def test_completed_gene(self):
        gene = 'gene2'
        with open(f'{self.temp_dir}/{gene}_estimator.pkl', 'w') as f:
            f.write('dummy data')
        self.assertEqual(len(self.queue), 4)

    def test_empty_queue(self):
        for gene in self.all_genes:
            with open(f'{self.temp_dir}/{gene}_estimator.pkl', 'w') as f:
                f.write('dummy data')
        self.assertTrue(self.queue.is_empty)
        self.assertRaises(StopIteration, next, self.queue)
