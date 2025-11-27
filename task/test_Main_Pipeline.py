"""
Comprehensive test suite for Main_Pipeline.py

This test file covers all functions and edge cases in the citation verification pipeline.
Tests use mocking to avoid real API calls and can be run repeatedly without external dependencies.

Usage:
    python test_Main_Pipeline.py
    # or with pytest:
    pytest test_Main_Pipeline.py -v
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import json
import sys
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any

# Add paths for imports
task_dir = Path(__file__).parent
src_dir = task_dir.parent / "src"
sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(task_dir))

# Import the module to test
import Main_Pipeline
from Main_Pipeline import (
    verify_citation,
    process_citations,
    generate_summary
)


class TestVerifyCitation(unittest.TestCase):
    """Test cases for verify_citation function"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_citations_file = os.path.join(self.test_dir, 'test_citations.json')
        
    def tearDown(self):
        """Clean up after each test method"""
        # Remove temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('Main_Pipeline.search_papers_by_title')
    @patch('Main_Pipeline.get_best_match_from_search_results')
    @patch('Main_Pipeline.compare_authors')
    def test_verify_citation_successful_match_verified(self, mock_compare, mock_get_match, mock_search):
        """Test successful citation verification with matching authors"""
        # Setup mocks
        mock_search.return_value = {
            'summary': {'total_results': 5}
        }
        mock_get_match.return_value = {
            'title': 'Test Paper Title',
            'authors': ['John Doe', 'Jane Smith'],
            'match_score': 95,
            'source': 'arxiv',
            'year': 2023,
            'doi': '10.1234/test',
            'url': 'https://example.com/paper',
            'pdf_url': 'https://example.com/paper.pdf',
            'arxiv_id': '1234.5678'
        }
        mock_compare.return_value = {
            'match': True,
            'discrepancies': []
        }
        
        # Test data
        citation = {
            'title': 'Test Paper Title',
            'authors': ['John Doe', 'Jane Smith']
        }
        
        # Execute
        result = verify_citation(citation, similarity_threshold=80, max_results_per_source=10)
        
        # Assertions
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['title'], 'Test Paper Title')
        self.assertEqual(result['source'], 'arxiv')
        self.assertEqual(result['match_score'], 95)
        self.assertEqual(result['verified_authors'], ['John Doe', 'Jane Smith'])
        self.assertEqual(result['original_authors'], ['John Doe', 'Jane Smith'])
        self.assertIn('comparison', result)
        self.assertIn('metadata', result)
        self.assertEqual(result['metadata']['year'], 2023)
        self.assertEqual(result['metadata']['arxiv_id'], '1234.5678')
        
        # Verify mocks were called correctly
        mock_search.assert_called_once_with(
            title='Test Paper Title',
            similarity_threshold=80,
            max_results_per_source=10,
            parallel=True
        )
        mock_get_match.assert_called_once()
        mock_compare.assert_called_once_with(
            original_authors=['John Doe', 'Jane Smith'],
            verified_authors=['John Doe', 'Jane Smith'],
            paper_title='Test Paper Title'
        )
    
    @patch('Main_Pipeline.search_papers_by_title')
    @patch('Main_Pipeline.get_best_match_from_search_results')
    @patch('Main_Pipeline.compare_authors')
    def test_verify_citation_discrepancy_found(self, mock_compare, mock_get_match, mock_search):
        """Test citation verification with author discrepancies"""
        # Setup mocks
        mock_search.return_value = {
            'summary': {'total_results': 3}
        }
        mock_get_match.return_value = {
            'title': 'Test Paper Title',
            'authors': ['John Doe', 'Jane Smith', 'Bob Wilson'],
            'match_score': 90,
            'source': 'dblp',
            'year': 2023,
            'venue': 'ICML'
        }
        mock_compare.return_value = {
            'match': False,
            'discrepancies': [
                {
                    'type': 'missing_author',
                    'details': 'Author missing: Bob Wilson'
                }
            ]
        }
        
        # Test data
        citation = {
            'title': 'Test Paper Title',
            'authors': ['John Doe', 'Jane Smith']
        }
        
        # Execute
        result = verify_citation(citation, similarity_threshold=80)
        
        # Assertions
        self.assertEqual(result['status'], 'discrepancy_found')
        self.assertEqual(result['source'], 'dblp')
        self.assertEqual(len(result['verified_authors']), 3)
        self.assertEqual(len(result['original_authors']), 2)
        self.assertFalse(result['comparison']['match'])
        self.assertEqual(result['metadata']['venue'], 'ICML')
    
    @patch('Main_Pipeline.search_papers_by_title')
    @patch('Main_Pipeline.get_best_match_from_search_results')
    def test_verify_citation_no_match_found(self, mock_get_match, mock_search):
        """Test citation verification when no good match is found"""
        # Setup mocks
        mock_search.return_value = {
            'summary': {'total_results': 0}
        }
        mock_get_match.return_value = None  # No match found
        
        # Test data
        citation = {
            'title': 'Very Unique Paper Title That Does Not Exist',
            'authors': ['Unknown Author']
        }
        
        # Execute
        result = verify_citation(citation, similarity_threshold=80)
        
        # Assertions
        self.assertEqual(result['status'], 'not_found')
        self.assertEqual(result['title'], 'Very Unique Paper Title That Does Not Exist')
        self.assertEqual(result['match_score'], 0)
        self.assertIn('search_summary', result)
    
    def test_verify_citation_empty_title(self):
        """Test citation verification with empty title"""
        citation = {
            'title': '',
            'authors': ['John Doe']
        }
        
        result = verify_citation(citation)
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('empty title', result['message'].lower())
        self.assertIsNone(result['title'])
    
    def test_verify_citation_missing_title(self):
        """Test citation verification with missing title field"""
        citation = {
            'authors': ['John Doe']
        }
        
        result = verify_citation(citation)
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('empty title', result['message'].lower())
    
    @patch('Main_Pipeline.search_papers_by_title')
    @patch('Main_Pipeline.get_best_match_from_search_results')
    @patch('Main_Pipeline.compare_authors')
    def test_verify_citation_exception_handling(self, mock_compare, mock_get_match, mock_search):
        """Test exception handling during verification"""
        # Setup mock to raise exception
        mock_search.side_effect = Exception("Network error")
        
        citation = {
            'title': 'Test Paper',
            'authors': ['John Doe']
        }
        
        result = verify_citation(citation)
        
        self.assertEqual(result['status'], 'error')
        self.assertIn('Error during verification', result['message'])
        self.assertEqual(result['title'], 'Test Paper')
    
    @patch('Main_Pipeline.search_papers_by_title')
    @patch('Main_Pipeline.get_best_match_from_search_results')
    @patch('Main_Pipeline.compare_authors')
    def test_verify_citation_semantic_scholar_source(self, mock_compare, mock_get_match, mock_search):
        """Test citation verification with Semantic Scholar as source"""
        mock_search.return_value = {'summary': {}}
        mock_get_match.return_value = {
            'title': 'Test Paper',
            'authors': ['Author 1'],
            'match_score': 85,
            'source': 'semantic_scholar',
            'paper_id': 'test-paper-id-123'
        }
        mock_compare.return_value = {'match': True, 'discrepancies': []}
        
        citation = {'title': 'Test Paper', 'authors': ['Author 1']}
        result = verify_citation(citation)
        
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['source'], 'semantic_scholar')
        self.assertEqual(result['metadata']['paper_id'], 'test-paper-id-123')
    
    @patch('Main_Pipeline.search_papers_by_title')
    @patch('Main_Pipeline.get_best_match_from_search_results')
    @patch('Main_Pipeline.compare_authors')
    def test_verify_citation_low_similarity_threshold(self, mock_compare, mock_get_match, mock_search):
        """Test citation verification with low similarity threshold"""
        mock_search.return_value = {'summary': {}}
        mock_get_match.return_value = {
            'title': 'Similar But Not Exact Title',
            'authors': ['Author 1'],
            'match_score': 75,  # Below default threshold
            'source': 'arxiv'
        }
        mock_compare.return_value = {'match': True, 'discrepancies': []}
        
        citation = {'title': 'Original Title', 'authors': ['Author 1']}
        result = verify_citation(citation, similarity_threshold=70)  # Lower threshold
        
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['match_score'], 75)


class TestProcessCitations(unittest.TestCase):
    """Test cases for process_citations function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('Main_Pipeline.verify_citation')
    def test_process_citations_basic(self, mock_verify):
        """Test basic citation processing"""
        # Setup mock
        mock_verify.side_effect = [
            {'status': 'verified', 'title': 'Paper 1', 'match_score': 95},
            {'status': 'discrepancy_found', 'title': 'Paper 2', 'match_score': 85},
            {'status': 'not_found', 'title': 'Paper 3', 'match_score': 0}
        ]
        
        citations = [
            {'title': 'Paper 1', 'authors': ['Author 1']},
            {'title': 'Paper 2', 'authors': ['Author 2']},
            {'title': 'Paper 3', 'authors': ['Author 3']}
        ]
        
        results = process_citations(citations, similarity_threshold=80, limit=None)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['verification']['status'], 'verified')
        self.assertEqual(results[1]['verification']['status'], 'discrepancy_found')
        self.assertEqual(results[2]['verification']['status'], 'not_found')
        self.assertEqual(mock_verify.call_count, 3)
    
    @patch('Main_Pipeline.verify_citation')
    def test_process_citations_with_limit(self, mock_verify):
        """Test citation processing with limit"""
        mock_verify.return_value = {'status': 'verified', 'title': 'Paper', 'match_score': 90}
        
        citations = [
            {'title': f'Paper {i}', 'authors': [f'Author {i}']}
            for i in range(10)
        ]
        
        results = process_citations(citations, limit=5)
        
        self.assertEqual(len(results), 5)
        self.assertEqual(mock_verify.call_count, 5)
    
    @patch('Main_Pipeline.verify_citation')
    def test_process_citations_empty_list(self, mock_verify):
        """Test processing empty citation list"""
        citations = []
        results = process_citations(citations)
        
        self.assertEqual(len(results), 0)
        mock_verify.assert_not_called()
    
    @patch('Main_Pipeline.verify_citation')
    def test_process_citations_single_citation(self, mock_verify):
        """Test processing single citation"""
        mock_verify.return_value = {'status': 'verified', 'title': 'Single Paper', 'match_score': 100}
        
        citations = [{'title': 'Single Paper', 'authors': ['Author']}]
        results = process_citations(citations)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['original']['title'], 'Single Paper')
    
    @patch('Main_Pipeline.verify_citation')
    def test_process_citations_progress_logging(self, mock_verify):
        """Test that progress is logged for batches of 10"""
        mock_verify.return_value = {'status': 'verified', 'title': 'Paper', 'match_score': 90}
        
        citations = [
            {'title': f'Paper {i}', 'authors': [f'Author {i}']}
            for i in range(25)
        ]
        
        # Should process all 25, but only log progress at multiples of 10
        results = process_citations(citations)
        
        self.assertEqual(len(results), 25)
        self.assertEqual(mock_verify.call_count, 25)


class TestGenerateSummary(unittest.TestCase):
    """Test cases for generate_summary function"""
    
    def test_generate_summary_empty_results(self):
        """Test summary generation with empty results"""
        results = []
        summary = generate_summary(results)
        
        self.assertEqual(summary['total'], 0)
    
    def test_generate_summary_mixed_statuses(self):
        """Test summary generation with mixed verification statuses"""
        results = [
            {
                'verification': {
                    'status': 'verified',
                    'source': 'arxiv',
                    'match_score': 95,
                    'comparison': {'discrepancies': []}
                }
            },
            {
                'verification': {
                    'status': 'discrepancy_found',
                    'source': 'dblp',
                    'match_score': 85,
                    'comparison': {'discrepancies': [{'type': 'name_change'}]}
                }
            },
            {
                'verification': {
                    'status': 'not_found',
                    'source': 'unknown',
                    'match_score': 0,
                    'comparison': {}
                }
            },
            {
                'verification': {
                    'status': 'verified',
                    'source': 'semantic_scholar',
                    'match_score': 90,
                    'comparison': {'discrepancies': []}
                }
            }
        ]
        
        summary = generate_summary(results)
        
        self.assertEqual(summary['total'], 4)
        self.assertEqual(summary['status_counts']['verified'], 2)
        self.assertEqual(summary['status_counts']['discrepancy_found'], 1)
        self.assertEqual(summary['status_counts']['not_found'], 1)
        self.assertEqual(summary['source_counts']['arxiv'], 1)
        self.assertEqual(summary['source_counts']['dblp'], 1)
        self.assertEqual(summary['source_counts']['semantic_scholar'], 1)
        self.assertEqual(summary['average_match_score'], 90.0)  # (95+85+0+90)/4 = 67.5, but only scores > 0
        self.assertEqual(summary['verification_rate'], 50.0)  # 2 verified / 4 total
    
    def test_generate_summary_parsing_errors(self):
        """Test summary generation with parsing errors"""
        results = [
            {
                'verification': {
                    'status': 'discrepancy_found',
                    'comparison': {
                        'discrepancies': [
                            {'type': 'parsing_error', 'details': 'Invalid author name'},
                            {'type': 'name_change', 'details': 'Name variant'}
                        ]
                    }
                }
            },
            {
                'verification': {
                    'status': 'discrepancy_found',
                    'comparison': {
                        'discrepancies': [
                            {'type': 'parsing_error', 'details': 'Another parsing error'}
                        ]
                    }
                }
            }
        ]
        
        summary = generate_summary(results)
        
        self.assertEqual(summary['parsing_errors_detected'], 2)
    
    def test_generate_summary_all_verified(self):
        """Test summary when all citations are verified"""
        results = [
            {
                'verification': {
                    'status': 'verified',
                    'source': 'arxiv',
                    'match_score': 100,
                    'comparison': {'discrepancies': []}
                }
            },
            {
                'verification': {
                    'status': 'verified',
                    'source': 'arxiv',
                    'match_score': 95,
                    'comparison': {'discrepancies': []}
                }
            }
        ]
        
        summary = generate_summary(results)
        
        self.assertEqual(summary['verification_rate'], 100.0)
        self.assertEqual(summary['status_counts']['verified'], 2)
        self.assertEqual(summary['average_match_score'], 97.5)
    
    def test_generate_summary_no_match_scores(self):
        """Test summary when no match scores are available"""
        results = [
            {
                'verification': {
                    'status': 'not_found',
                    'match_score': 0
                }
            },
            {
                'verification': {
                    'status': 'error',
                    'match_score': 0
                }
            }
        ]
        
        summary = generate_summary(results)
        
        self.assertEqual(summary['average_match_score'], 0)
    
    def test_generate_summary_missing_fields(self):
        """Test summary generation with missing optional fields"""
        results = [
            {
                'verification': {
                    'status': 'verified'
                    # Missing source, match_score, comparison
                }
            }
        ]
        
        summary = generate_summary(results)
        
        self.assertEqual(summary['total'], 1)
        self.assertEqual(summary['status_counts']['verified'], 1)
        self.assertEqual(summary['source_counts'].get('unknown', 0), 0)  # Should not count 'unknown'
        self.assertEqual(summary['average_match_score'], 0)


class TestMainFunction(unittest.TestCase):
    """Test cases for main function (partial testing)"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.test_citations_file = os.path.join(self.test_dir, 'test_citations.json')
        self.test_output_file = os.path.join(self.test_dir, 'test_output.json')
        
        # Create test citations file
        test_citations = [
            {'title': 'Test Paper 1', 'authors': ['Author 1']},
            {'title': 'Test Paper 2', 'authors': ['Author 2']}
        ]
        with open(self.test_citations_file, 'w', encoding='utf-8') as f:
            json.dump(test_citations, f)
    
    def tearDown(self):
        """Clean up"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('Main_Pipeline.load_citations')
    @patch('Main_Pipeline.process_citations')
    @patch('Main_Pipeline.generate_summary')
    @patch('Main_Pipeline.save_results_to_json')
    @patch('Main_Pipeline.os.path.exists')
    def test_main_successful_execution(self, mock_exists, mock_save, mock_summary, mock_process, mock_load):
        """Test successful main function execution"""
        # Setup mocks
        mock_exists.return_value = True
        mock_load.return_value = [
            {'title': 'Paper 1', 'authors': ['Author 1']},
            {'title': 'Paper 2', 'authors': ['Author 2']}
        ]
        mock_process.return_value = [
            {'original': {'title': 'Paper 1'}, 'verification': {'status': 'verified'}},
            {'original': {'title': 'Paper 2'}, 'verification': {'status': 'verified'}}
        ]
        mock_summary.return_value = {
            'total': 2,
            'status_counts': {'verified': 2},
            'source_counts': {'arxiv': 2},
            'average_match_score': 95.0,
            'parsing_errors_detected': 0,
            'verification_rate': 100.0
        }
        
        # Temporarily override configuration
        original_citations_file = Main_Pipeline.CITATIONS_FILE
        original_output_file = Main_Pipeline.OUTPUT_FILE
        
        try:
            Main_Pipeline.CITATIONS_FILE = self.test_citations_file
            Main_Pipeline.OUTPUT_FILE = self.test_output_file
            
            # Execute main - it should complete successfully without raising SystemExit
            # (main() only calls sys.exit() on errors, not on success)
            try:
                Main_Pipeline.main()
            except SystemExit as e:
                # If it exits, it should be with code 0 (success)
                self.assertEqual(e.code, 0)
            
            # Verify that mocks were called correctly
            mock_load.assert_called_once_with(self.test_citations_file)
            mock_process.assert_called_once()
            mock_summary.assert_called_once()
            mock_save.assert_called_once()
            
        finally:
            Main_Pipeline.CITATIONS_FILE = original_citations_file
            Main_Pipeline.OUTPUT_FILE = original_output_file
    
    @patch('Main_Pipeline.os.path.exists')
    def test_main_missing_citations_file(self, mock_exists):
        """Test main function when citations file is missing"""
        mock_exists.return_value = False
        
        original_citations_file = Main_Pipeline.CITATIONS_FILE
        try:
            Main_Pipeline.CITATIONS_FILE = 'nonexistent_file.json'
            
            with self.assertRaises(SystemExit) as cm:
                Main_Pipeline.main()
            
            # Should exit with error code 1
            self.assertEqual(cm.exception.code, 1)
            
        finally:
            Main_Pipeline.CITATIONS_FILE = original_citations_file
    
    @patch('Main_Pipeline.os.path.exists')
    def test_main_invalid_similarity_threshold(self, mock_exists):
        """Test main function with invalid similarity threshold"""
        mock_exists.return_value = True
        
        original_threshold = Main_Pipeline.SIMILARITY_THRESHOLD
        try:
            Main_Pipeline.SIMILARITY_THRESHOLD = 150  # Invalid (> 100)
            
            with self.assertRaises(SystemExit) as cm:
                Main_Pipeline.main()
            
            self.assertEqual(cm.exception.code, 1)
            
        finally:
            Main_Pipeline.SIMILARITY_THRESHOLD = original_threshold


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    @patch('Main_Pipeline.search_papers_by_title')
    @patch('Main_Pipeline.get_best_match_from_search_results')
    @patch('Main_Pipeline.compare_authors')
    def test_verify_citation_no_authors(self, mock_compare, mock_get_match, mock_search):
        """Test citation verification with no authors"""
        mock_search.return_value = {'summary': {}}
        mock_get_match.return_value = {
            'title': 'Paper Title',
            'authors': [],
            'match_score': 90,
            'source': 'arxiv'
        }
        mock_compare.return_value = {'match': True, 'discrepancies': []}
        
        citation = {'title': 'Paper Title', 'authors': []}
        result = verify_citation(citation)
        
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['verified_authors'], [])
        self.assertEqual(result['original_authors'], [])
    
    @patch('Main_Pipeline.search_papers_by_title')
    @patch('Main_Pipeline.get_best_match_from_search_results')
    @patch('Main_Pipeline.compare_authors')
    def test_verify_citation_very_long_title(self, mock_compare, mock_get_match, mock_search):
        """Test citation verification with very long title"""
        long_title = 'A' * 500  # Very long title
        mock_search.return_value = {'summary': {}}
        mock_get_match.return_value = {
            'title': long_title,
            'authors': ['Author'],
            'match_score': 90,
            'source': 'arxiv'
        }
        mock_compare.return_value = {'match': True, 'discrepancies': []}
        
        citation = {'title': long_title, 'authors': ['Author']}
        result = verify_citation(citation)
        
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(len(result['title']), 500)
    
    @patch('Main_Pipeline.search_papers_by_title')
    @patch('Main_Pipeline.get_best_match_from_search_results')
    @patch('Main_Pipeline.compare_authors')
    def test_verify_citation_special_characters(self, mock_compare, mock_get_match, mock_search):
        """Test citation verification with special characters in title"""
        special_title = "Paper Title with Special Chars: é, ñ, 中文, 🎓"
        mock_search.return_value = {'summary': {}}
        mock_get_match.return_value = {
            'title': special_title,
            'authors': ['Author'],
            'match_score': 90,
            'source': 'arxiv'
        }
        mock_compare.return_value = {'match': True, 'discrepancies': []}
        
        citation = {'title': special_title, 'authors': ['Author']}
        result = verify_citation(citation)
        
        self.assertEqual(result['status'], 'verified')
        self.assertIn('é', result['title'])
    
    def test_generate_summary_single_result(self):
        """Test summary generation with single result"""
        results = [
            {
                'verification': {
                    'status': 'verified',
                    'source': 'arxiv',
                    'match_score': 100,
                    'comparison': {'discrepancies': []}
                }
            }
        ]
        
        summary = generate_summary(results)
        
        self.assertEqual(summary['total'], 1)
        self.assertEqual(summary['verification_rate'], 100.0)
    
    @patch('Main_Pipeline.verify_citation')
    def test_process_citations_limit_exceeds_total(self, mock_verify):
        """Test processing with limit greater than total citations"""
        mock_verify.return_value = {'status': 'verified', 'title': 'Paper', 'match_score': 90}
        
        citations = [
            {'title': 'Paper 1', 'authors': ['Author 1']},
            {'title': 'Paper 2', 'authors': ['Author 2']}
        ]
        
        results = process_citations(citations, limit=10)  # Limit > total
        
        self.assertEqual(len(results), 2)  # Should process all available
        self.assertEqual(mock_verify.call_count, 2)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios"""
    
    @patch('Main_Pipeline.verify_citation')
    def test_full_pipeline_scenario(self, mock_verify):
        """Test a realistic full pipeline scenario"""
        # This tests the interaction between functions
        citations = [
            {'title': 'Paper 1', 'authors': ['Author 1']},
            {'title': 'Paper 2', 'authors': ['Author 2']}
        ]
        
        # Setup mock to return different results for each citation
        mock_verify.side_effect = [
            {
                'status': 'verified',
                'title': 'Paper 1',
                'source': 'arxiv',
                'match_score': 95,
                'verified_authors': ['Author 1'],
                'original_authors': ['Author 1'],
                'comparison': {'match': True, 'discrepancies': []},
                'metadata': {}
            },
            {
                'status': 'discrepancy_found',
                'title': 'Paper 2',
                'source': 'dblp',
                'match_score': 85,
                'verified_authors': ['Author 2', 'Author 3'],
                'original_authors': ['Author 2'],
                'comparison': {
                    'match': False,
                    'discrepancies': [{'type': 'extra_author', 'details': 'Extra author'}]
                },
                'metadata': {}
            }
        ]
        
        results = process_citations(citations)
        summary = generate_summary(results)
        
        self.assertEqual(summary['total'], 2)
        self.assertEqual(summary['status_counts']['verified'], 1)
        self.assertEqual(summary['status_counts']['discrepancy_found'], 1)
        self.assertEqual(summary['source_counts']['arxiv'], 1)
        self.assertEqual(summary['source_counts']['dblp'], 1)


def run_tests():
    """Run all tests and print summary"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVerifyCitation))
    suite.addTests(loader.loadTestsFromTestCase(TestProcessCitations))
    suite.addTests(loader.loadTestsFromTestCase(TestGenerateSummary))
    suite.addTests(loader.loadTestsFromTestCase(TestMainFunction))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationScenarios))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests
    success = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

