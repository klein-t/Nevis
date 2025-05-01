import os
import json
import re
import glob
from typing import Dict, Any, List, Tuple, Set, Union
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class TargetedJSONComparison:
    """
    A targeted comparison algorithm for financial profile JSON data that applies
    domain-specific rules and data-type-based normalization to measure semantic similarity.
    """

    def __init__(self, ground_truth: Dict, extracted: Dict):
        """Initialize with the two JSONs to compare"""
        self.ground_truth = ground_truth
        self.extracted = extracted
        self.results = {
            "matches": 0,
            "partial_matches": 0,
            "mismatches": 0,
            "missing": 0,
            "total": 0,
            "details": []
        }
        self.category_metrics = {}
        # Define entity resolution map early for use in processors
        self.entity_map = {
            "client": "person",
            "individual": "person",
        }
        # Define data-type based processors
        self.processors = self._define_processors()
        # Define field-specific comparison rules
        self.field_rules = self._define_field_rules()

    def compare(self) -> Dict:
        """Perform the comparison using domain-specific rules"""
        # Perform the comparison using the rules defined in __init__
        self._compare_fields()

        # Calculate overall metrics
        self._calculate_metrics()

        return {
            "results": self.results,
            "accuracy": self.accuracy,
            "category_metrics": self.category_metrics
        }

    def _define_processors(self) -> Dict:
        """Define data-type based processors instead of field-specific ones"""
        return {
            # Basic values (strings, simple fields) - mapped to 'normalize_value' key
            "normalize_value": lambda value: self._normalize_basic_value(value),

            # Dates
            "normalize_date": lambda value: self._normalize_date_value(value),

            # Amounts/numeric values
            "normalize_amount": lambda value: self._normalize_numeric_value(value),

            # Text with semantic meaning
            "normalize_text": lambda value: self._normalize_text_value(value),

            # Entity references (owners, clients)
            "normalize_owner": lambda value: self._normalize_entity_reference(value),

            # Special case for amount_assured (now folded into normalize_amount)
            "normalize_amount_assured": lambda value: self._normalize_numeric_value(value)
        }

    def _normalize_basic_value(self, value):
        """Normalize basic values (strings, simple fields)"""
        if value is None or value == "":
            return ""
        return str(value).lower()

    def _normalize_date_value(self, value):
        """Normalize date values to consistent format (YYYY-MM)"""
        if not value:
            return ""
        # Handle both string dates and other formats
        value_str = str(value)
        # Extract just year and month for comparison
        match = re.match(r'^(\d{4})-(\d{2})', value_str)
        return f"{match[1]}-{match[2]}" if match else value_str.lower()

    def _normalize_numeric_value(self, value):
        """Normalize numeric values for comparison"""
        if value is None or value == "":
             return "" # Or perhaps None is better here? Let's stick with "" for now.

        # Handle potential sums in strings (like amount_assured case)
        if isinstance(value, str) and ',' in value:
            try:
                # Attempt to sum comma-separated numbers
                return sum(float(a.strip()) for a in value.split(','))
            except (ValueError, TypeError):
                # If summing fails, fall back to basic parsing
                pass

        # Try to convert to float if possible
        try:
            return float(value)
        except (ValueError, TypeError):
            # If direct conversion fails, return normalized string representation
            return str(value).lower()


    def _normalize_text_value(self, value):
        """Normalize text with semantic meaning"""
        if not value:
            return ""
        # Convert to lower case, remove punctuation, standardize whitespace
        return re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()]', ' ', str(value).lower()).replace('\n', ' ').replace('\t', ' ').replace('\r', ' ').replace('  ', ' ').strip()

    def _normalize_entity_reference(self, value):
        """Normalize entity references with identity resolution"""
        if not value:
            return ""
        value_lower = str(value).lower()
        # Use the entity_map defined in __init__
        return self.entity_map.get(value_lower, value_lower)

    def _define_field_rules(self) -> Dict:
        """Define field-specific comparison rules based on data types"""
        field_rules = {
            # Personal Details
            "personal_details.client.title": {
                "process": self.processors["normalize_value"],
                "description": "Title",
                "required": False
            },
            "personal_details.client.first_name": {
                "process": self.processors["normalize_value"],
                "description": "First Name",
                "required": True
            },
            "personal_details.client.middle_names": {
                "process": self.processors["normalize_value"],
                "description": "Middle Names",
                "required": False
            },
            "personal_details.client.last_name": {
                "process": self.processors["normalize_value"],
                "description": "Last Name",
                "required": True
            },
            "personal_details.client.known_as": {
                "process": self.processors["normalize_value"],
                "description": "Known As",
                "required": False
            },
            "personal_details.client.pronouns": {
                "process": self.processors["normalize_value"],
                "description": "Pronouns",
                "required": False
            },
            "personal_details.client.date_of_birth": {
                "process": self.processors["normalize_date"],
                "description": "Date of Birth",
                "required": True
            },
            "personal_details.client.place_of_birth": {
                "process": self.processors["normalize_text"],
                "description": "Place of Birth",
                "required": False,
                # Match if GT city name is within EX text
                "isMatch": lambda gt, ex: gt.split(',')[0].strip() in ex if gt and ex else (True if not gt and not ex else False)
            },
            "personal_details.client.nationality": {
                "process": self.processors["normalize_text"],
                "description": "Nationality",
                "required": False,
                 # Match if GT nationality is within EX text
                "isMatch": lambda gt, ex: gt in ex if gt and ex else (True if not gt and not ex else False)
            },
            "personal_details.client.gender": {
                "process": self.processors["normalize_value"],
                "description": "Gender",
                "required": False
            },
            "personal_details.client.marital_status": {
                "process": self.processors["normalize_value"],
                "description": "Marital Status",
                "required": False
            },
            "personal_details.client.email_address": {
                "process": self.processors["normalize_value"],
                "description": "Email Address",
                "required": False
            },
            "personal_details.client.mobile_phone": {
                "process": self.processors["normalize_value"],
                "description": "Mobile Phone",
                "required": False
            },

            # Address
            "personal_details.current_address.ownership_status": {
                "process": self.processors["normalize_value"],
                "description": "Ownership Status",
                "required": False
            },
            "personal_details.current_address.postcode": {
                "process": self.processors["normalize_value"],
                "description": "Postcode",
                "required": False
            },
            "personal_details.current_address.town_city": {
                "process": self.processors["normalize_value"],
                "description": "Town/City",
                "required": False
            },
            "personal_details.current_address.move_in_date": {
                "process": self.processors["normalize_date"],
                "description": "Move-in Date",
                "required": False
            },

            # Employment
            "employment.client.country_domiciled": {
                "process": self.processors["normalize_value"],
                "description": "Country Domiciled",
                "required": False
            },
            "employment.client.employment_status": {
                "process": self.processors["normalize_value"],
                "description": "Employment Status",
                "required": False
            },
            "employment.client.occupation": {
                "process": self.processors["normalize_value"],
                "description": "Occupation",
                "required": False
            },
            "employment.client.employer": {
                "process": self.processors["normalize_value"],
                "description": "Employer",
                "required": False
            },
            "employment.client.employment_started": {
                "process": self.processors["normalize_date"],
                "description": "Employment Start Date",
                "required": False
            },

            # Income (Assuming incomes is a list, comparison might need adjustment for list handling)
            # Rule below assumes a single income object or applies to *each* income item if handled in _compare_fields
            # Need logic to handle comparing lists of objects if 'incomes' is a list.
            # These rules would apply *within* each matched income item.
            "incomes.owner": {
                "process": self.processors["normalize_owner"],
                "description": "Income Owner",
                "required": False
            },
            "incomes.name": {
                "process": self.processors["normalize_text"],
                "description": "Income Name",
                "required": False,
                "isMatch": lambda gt, ex: "salary" in gt and "salary" in ex if gt and ex else gt == ex
            },
            "incomes.amount": {
                "process": self.processors["normalize_amount"],
                "description": "Income Amount",
                "required": False
            },
            "incomes.frequency": {
                "process": self.processors["normalize_value"], 
                "description": "Income Frequency",
                "required": False
            },

            "expenses.loan_repayments.amount": {
                "process": self.processors["normalize_amount"],
                "description": "Loan Repayment Amount",
                "required": False
            },
            "expenses.housing_expenses.amount": {
                "process": self.processors["normalize_amount"],
                "description": "Housing Expense Amount",
                "required": False
            },

            "pensions.type": {
                "process": self.processors["normalize_value"],
                "description": "Pension Type",
                "required": False
            },
            "pensions.provider": {
                "process": self.processors["normalize_value"],
                "description": "Pension Provider",
                "required": False
            },
            "pensions.value": {
                "process": self.processors["normalize_amount"],
                "description": "Pension Value",
                "required": False
            },

            "loans_mortgages.type": {
                "process": self.processors["normalize_value"],
                "description": "Mortgage Type",
                "required": False,
                "isMatch": lambda gt, ex: "fixed" in gt and "fixed" in ex if gt and ex else gt == ex
            },
            "loans_mortgages.monthly_cost": {
                "process": self.processors["normalize_amount"],
                "description": "Monthly Cost",
                "required": False
            },
            "loans_mortgages.outstanding_value": {
                "process": self.processors["normalize_amount"],
                "description": "Outstanding Value",
                "required": False
            },
            "loans_mortgages.interest_rate": {
                "process": self.processors["normalize_amount"],
                "description": "Interest Rate",
                "required": False
            },

     
            "health_details.client.current_state_of_health": {
                "process": self.processors["normalize_value"],
                "description": "Current Health",
                "required": False
            },
            "health_details.client.smoker": {
                "process": self.processors["normalize_value"],
                "description": "Smoker",
                "required": False
            },

            "protection_policies.type": {
                "process": self.processors["normalize_text"],
                "description": "Protection Type",
                "required": False,
                "isMatch": lambda gt, ex: "life" in gt and "life" in ex if gt and ex else gt == ex
            },
            "protection_policies.provider": {
                "process": self.processors["normalize_text"],
                "description": "Protection Provider",
                "required": False,
                "isMatch": lambda gt, ex: "life insurance" in gt and "life insurance" in ex if gt and ex else gt == ex
            },
            "protection_policies.monthly_cost": {
                "process": self.processors["normalize_amount"],
                "description": "Monthly Premium",
                "required": False
            },
            "protection_policies.amount_assured": {

                "process": self.processors["normalize_amount_assured"],
                "description": "Amount Assured",
                "required": False
            }
        }

        return field_rules



    def _compare_fields(self):
        """Compare fields according to the defined rules"""
        # Use field rules defined in __init__
        # If rules are empty (e.g., not defined or loaded), auto-generate them
        if not self.field_rules:
            print("Warning: No field rules defined. Attempting to auto-generate.")
            self._auto_generate_field_rules()
            if not self.field_rules:
                print("Error: Could not auto-generate field rules. Comparison cannot proceed.")
                return # Cannot compare without rules

        # Gather all categories for reporting
        categories = {}

        # Process each field according to its rules

        processed_paths = set() # Keep track of paths processed to avoid duplicates if auto-gen creates overlapping rules

        for field_path, rule in self.field_rules.items():
            if field_path in processed_paths:
                continue
            processed_paths.add(field_path)

            # Get category
            category = field_path.split('.')[0]
            if category not in categories:
                categories[category] = {"total": 0, "matched": 0, "partial_matched": 0, "mismatched": 0, "missing_gt": 0, "missing_ex": 0}
            categories[category]["total"] += 1
            self.results["total"] += 1

            # Get values from both JSONs
            gt_value = self._get_nested_value(self.ground_truth, field_path)
            ex_value = self._get_nested_value(self.extracted, field_path)

            # Process the values according to the field's rule
            process_func = rule.get("process", self.processors["normalize_value"])
            processed_gt = process_func(gt_value)
            processed_ex = process_func(ex_value)

            # Determine match status and similarity
            match_status = "mismatch" # Default status
            similarity = 0.0
            is_match = False
            is_partial = False

            required = rule.get("required", False)

            if gt_value is None and ex_value is None:
                 # Both missing - consider match only if not required
                 if not required:
                     match_status = "match"
                     similarity = 1.0
                     is_match = True
                 else:
                     match_status = "missing_both_required" # A specific type of mismatch
                     similarity = 0.0
                     self.results["missing"] += 1 # Count as missing if required
                     categories[category]["missing_gt"] += 1 # Arguably missing from both
                     categories[category]["missing_ex"] += 1
            elif gt_value is None:
                 # Missing in ground truth, present in extracted
                 match_status = "missing_gt"
                 similarity = 0.0
                 self.results["missing"] += 1
                 categories[category]["missing_gt"] += 1
            elif ex_value is None:
                 # Present in ground truth, missing in extracted
                 match_status = "missing_ex"
                 similarity = 0.0
                 self.results["missing"] += 1
                 categories[category]["missing_ex"] += 1
            else:
                 # Both have values, perform comparison
                 if "isMatch" in rule:
                     # Use custom matching function if provided
                     try:
                         is_match = rule["isMatch"](processed_gt, processed_ex)
                         similarity = 1.0 if is_match else 0.0
                         match_status = "match" if is_match else "mismatch"
                     except Exception as e:
                         print(f"Error in custom isMatch for {field_path}: {e}")
                         is_match = False
                         similarity = 0.0
                         match_status = "mismatch_error"
                 else:
                     # Standard comparison based on processed values
                     try:
                         # Handle numeric comparison with tolerance? For now, exact match after normalization.
                         if isinstance(processed_gt, (int, float)) and isinstance(processed_ex, (int, float)):
                             # Allow for small floating point differences
                             if abs(processed_gt - processed_ex) < 1e-6:
                                 is_match = True
                                 similarity = 1.0
                                 match_status = "match"
                             else:
                                 is_match = False
                                 similarity = 0.0 # Or calculate relative difference?
                                 match_status = "mismatch"
                         # Handle text comparison using similarity
                         elif isinstance(processed_gt, str) and isinstance(processed_ex, str):
                             similarity = self._calculate_text_similarity(processed_gt, processed_ex)
                             if similarity >= 0.95: # High threshold for full match
                                 is_match = True
                                 match_status = "match"
                             elif similarity >= 0.5: # Lower threshold for partial match
                                 is_partial = True
                                 match_status = "partial_match"
                             else:
                                 is_match = False
                                 match_status = "mismatch"
                         # Default: Exact equality check
                         else:
                             if processed_gt == processed_ex:
                                 is_match = True
                                 similarity = 1.0
                                 match_status = "match"
                             else:
                                 is_match = False
                                 similarity = 0.0
                                 match_status = "mismatch"

                     except TypeError:
                         # Handle potential comparison errors between incompatible types after processing
                         is_match = False
                         similarity = 0.0
                         match_status = "mismatch_type_error"


            # Update overall and category counters based on final status
            if match_status == "match":
                self.results["matches"] += 1
                categories[category]["matched"] += 1
            elif match_status == "partial_match":
                self.results["partial_matches"] += 1
                categories[category]["partial_matched"] += 1
            else: # All other statuses are considered mismatches for counts
                self.results["mismatches"] += 1
                categories[category]["mismatched"] += 1


            # Store detailed results
            self.results["details"].append({
                "field": field_path,
                "description": rule.get("description", field_path.split('.')[-1].replace('_', ' ').title()),
                "ground_truth": gt_value,
                "extracted": ex_value,
                "processed_ground_truth": processed_gt,
                "processed_extracted": processed_ex,
                "status": match_status, # Use detailed status
                "is_match": is_match or is_partial, # Combined flag for overall success
                "is_exact_match": is_match,
                "is_partial_match": is_partial,
                "similarity": round(similarity, 3),
                "category": category,
                "required": required
            })

        # Store category information for metrics calculation
        self.categories = categories


    def _auto_generate_field_rules(self):
        """Automatically generate field rules from the data if none are provided"""
        self.field_rules = {}
        processed_paths = set()

        def process_obj(obj, prefix="", depth=0):
            if depth > 5 or obj is None: # Limit recursion depth and handle None
                return

            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    if new_prefix in processed_paths: continue

                    # Basic heuristic: Don't generate rules for complex objects/lists themselves, only primitives
                    if isinstance(value, (dict, list)):
                         # Recurse only if it's not empty or just contains primitives
                         if value and not all(isinstance(item, (dict, list)) for item in (value if isinstance(value, list) else value.values())):
                             process_obj(value, new_prefix, depth + 1)
                         # Skip adding a rule for the container itself
                    else:
                        # Add a rule for this primitive field
                        if new_prefix not in self.field_rules:
                            processor_key = self._determine_processor(key, value)
                            self.field_rules[new_prefix] = {
                                "process": self.processors[processor_key],
                                "description": key.replace('_', ' ').title(),
                                "required": False # Default to not required
                            }
                            processed_paths.add(new_prefix)

            elif isinstance(obj, list) and obj:
                 first_item = obj[0]
                 # Only recurse if list contains dicts
                 if isinstance(first_item, dict):
                     process_obj(first_item, prefix, depth + 1) # Pass the same prefix for fields *within* the list items


        # Process both ground truth and extracted data to capture all possible fields
        print("Auto-generating rules from ground truth...")
        process_obj(self.ground_truth)
        print(f"Generated {len(self.field_rules)} rules from ground truth.")
        print("Auto-generating rules from extracted data...")
        start_count = len(self.field_rules)
        process_obj(self.extracted)
        print(f"Added {len(self.field_rules) - start_count} rules from extracted data.")
        print(f"Total auto-generated rules: {len(self.field_rules)}")


    def _determine_processor(self, key, value) -> str:
        """Determine the appropriate processor KEY based on field name and value type"""
        key_lower = key.lower()

        # Date fields
        if 'date' in key_lower or 'dob' in key_lower or 'started' in key_lower or 'payment_date' in key_lower:
            return "normalize_date"

        # Owner fields
        if key_lower == 'owner':
            return "normalize_owner"

        # Amount fields (needs careful check to avoid non-numeric like 'amount_description')
        if ('amount' in key_lower or 'cost' in key_lower or 'value' in key_lower or 'premium' in key_lower or 'salary' in key_lower or 'income' in key_lower or 'expense' in key_lower) and not key_lower.endswith('description'):
             # Further check if value looks numeric (even if string)
             try:
                 float(value)
                 return "normalize_amount"
             except (ValueError, TypeError, AttributeError):
                 # If it's not easily convertible, maybe it's just text
                 pass # Fall through to text/value checks

        # Specific complex cases identified in manual rules
        if key_lower == 'amount_assured':
            return "normalize_amount_assured" # Keep specific key if needed, maps to numeric

        # Text fields (longer strings)
        if isinstance(value, str) and len(value.split()) > 3: # Arbitrary threshold for 'longer' text
            return "normalize_text"

        # Entity/Name fields that aren't owners - use basic normalization
        if 'name' in key_lower or 'employer' in key_lower or 'provider' in key_lower or 'occupation' in key_lower:
             return "normalize_value" # Basic lowercase is usually sufficient

        # Categorical fields - use basic normalization
        if 'status' in key_lower or 'type' in key_lower or 'frequency' in key_lower or 'gender' in key_lower or 'priority' in key_lower:
             return "normalize_value"

        # Default for everything else (shorter strings, booleans, etc.)
        return "normalize_value"


    def _get_nested_value(self, obj, path):
        """Get a value from a nested object using dot notation path"""
        try:
            parts = path.split('.')
            current = obj
            for part in parts:
                if current is None:
                    return None
                # Handle list indexing if needed? Not currently supported by path format.
                # Example: 'incomes.0.amount' would require parsing '0' as index.
                if isinstance(current, dict):
                    current = current.get(part, None) # Use .get for safer access
                elif isinstance(current, list):
                     print(f"Warning: Trying to access part '{part}' on a list at path '{path}'. Returning None.")
                     return None # Cannot access dict key on a list
                else:
                    # Trying to access a part on a primitive value
                    return None
            return current
        except Exception as e:
            print(f"Error getting nested value for path {path}: {e}")
            return None


    def _calculate_text_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts using Jaccard index on words"""
        if not text1 and not text2: # Both empty
            return 1.0
        if not text1 or not text2: # One empty
            return 0.0

        # Use the processed text (already lowercased, punctuation removed)
        stop_words = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
                      "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
                      "to", "was", "were", "will", "with", "the", "this", "these", "those"}

        words1 = set(w for w in text1.split() if w not in stop_words and len(w) > 1)
        words2 = set(w for w in text2.split() if w not in stop_words and len(w) > 1)

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        if union == 0:
             # This happens if both texts consist only of stop words or short words
             # Check if original non-processed texts were identical in this edge case
             return 1.0 if text1 == text2 else 0.0 # Revert to exact match on normalized originals

        return intersection / union


    def _calculate_metrics(self):
        """Calculate overall and category metrics"""
        total = self.results["total"]
        if total > 0:
            # Accuracy considers both exact and partial matches as success
            self.accuracy = (self.results["matches"] + self.results["partial_matches"]) / total
        else:
            self.accuracy = 0

        # Calculate category metrics
        self.category_metrics = {}
        if not hasattr(self, 'categories'): # Ensure categories dict exists
             print("Warning: 'categories' attribute not found during metric calculation.")
             self.categories = {}

        for category, counts in self.categories.items():
            cat_total = counts.get("total", 0)
            cat_matched = counts.get("matched", 0)
            cat_partial = counts.get("partial_matched", 0)
            cat_mismatched = counts.get("mismatched", 0)
            # Include missing counts? Might be useful.
            # cat_missing_gt = counts.get("missing_gt", 0)
            # cat_missing_ex = counts.get("missing_ex", 0)

            if cat_total > 0:
                cat_accuracy = (cat_matched + cat_partial) / cat_total
                self.category_metrics[category] = {
                    "accuracy": round(cat_accuracy, 4),
                    "total_fields": cat_total,
                    "matched_fields": cat_matched,
                    "partial_matched_fields": cat_partial,
                    "mismatched_fields": cat_mismatched,
                    # Add missing counts if desired:
                    # "missing_ground_truth": cat_missing_gt,
                    # "missing_extracted": cat_missing_ex,
                }
            else:
                self.category_metrics[category] = {
                    "accuracy": 0,
                    "total_fields": 0,
                    "matched_fields": 0,
                    "partial_matched_fields": 0,
                    "mismatched_fields": 0,
                }

        # Sort details by category and field for better readability
        if self.results["details"]:
            self.results["details"].sort(key=lambda x: (x["category"], x["field"]))



def find_case_directories(base_dir: str) -> List[str]:
    """Find all case directories in the base directory"""
    pattern = os.path.join(base_dir, "case_*")
    case_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    case_dirs.sort(key=lambda d: int(re.search(r'case_0*(\d+)', os.path.basename(d)).group(1)))
    return case_dirs


def process_case(case_dir: str) -> Dict:
    """Process a single case directory"""
    ground_truth_path = os.path.join(case_dir, "ground_truth.json")
    extracted_data_path = os.path.join(case_dir, "extracted_data.json")

    if not os.path.exists(ground_truth_path) or not os.path.exists(extracted_data_path):
        print(f"Warning: Missing files in {case_dir}. Skipping.")
        return None

    try:
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        with open(extracted_data_path, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)

        # Instantiate and run comparison
        comparison = TargetedJSONComparison(ground_truth, extracted_data)
        result = comparison.compare()

        # Add case information
        case_name = os.path.basename(case_dir)
        result["case_name"] = case_name
        try:
            result["case_number"] = int(re.search(r'case_0*(\d+)', case_name).group(1))
        except (AttributeError, ValueError):
             result["case_number"] = -1 # Assign default if parsing fails
             print(f"Warning: Could not parse case number from directory name {case_name}")


        return result

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {case_dir}: {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing {case_dir}: {str(e)}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return None


def generate_summary_report(results: List[Dict]) -> Dict:
    """Generate a summary report from all case results"""
    if not results:
        return {"error": "No results to summarize"}

    valid_results = [r for r in results if r is not None and "results" in r] # Ensure result is valid

    if not valid_results:
        return {"error": "No valid results found to summarize"}

    # Aggregate overall metrics
    total_matches = sum(r["results"].get("matches", 0) for r in valid_results)
    total_partial_matches = sum(r["results"].get("partial_matches", 0) for r in valid_results)
    total_mismatches = sum(r["results"].get("mismatches", 0) for r in valid_results)
    total_missing = sum(r["results"].get("missing", 0) for r in valid_results) # This counts fields missing in EX when present in GT or vice-versa
    total_fields = sum(r["results"].get("total", 0) for r in valid_results)

    overall_accuracy = (total_matches + total_partial_matches) / total_fields if total_fields > 0 else 0

    # Per-case metrics
    case_metrics = []
    for result in valid_results:
         res_data = result.get("results", {})
         total = res_data.get("total", 0)
         accuracy = result.get("accuracy", 0) # Use pre-calculated accuracy

         case_metrics.append({
             "case_name": result.get("case_name", "Unknown"),
             "case_number": result.get("case_number", -1),
             "accuracy": accuracy,
             "matches": res_data.get("matches", 0),
             "partial_matches": res_data.get("partial_matches", 0),
             "mismatches": res_data.get("mismatches", 0),
             "missing": res_data.get("missing", 0),
             "total": total
         })

    case_metrics.sort(key=lambda m: m["case_number"])

    # Aggregate category metrics
    all_categories = set()
    for result in valid_results:
         if "category_metrics" in result:
             all_categories.update(result["category_metrics"].keys())

    category_summary = {}
    for category in sorted(list(all_categories)):
         cat_matches = sum(r.get("category_metrics", {}).get(category, {}).get("matched_fields", 0) for r in valid_results)
         cat_partial = sum(r.get("category_metrics", {}).get(category, {}).get("partial_matched_fields", 0) for r in valid_results)
         cat_total = sum(r.get("category_metrics", {}).get(category, {}).get("total_fields", 0) for r in valid_results)
         cat_mismatch = sum(r.get("category_metrics", {}).get(category, {}).get("mismatched_fields", 0) for r in valid_results) # Aggregate mismatches too

         cat_accuracy = (cat_matches + cat_partial) / cat_total if cat_total > 0 else 0

         category_summary[category] = {
             "accuracy": round(cat_accuracy, 4),
             "matches": cat_matches,
             "partial_matches": cat_partial,
             "mismatches": cat_mismatch,
             "total": cat_total
         }

    # Calculate common mismatches (including partials and missing)
    mismatch_fields = {}
    for result in valid_results:
         details = result.get("results", {}).get("details", [])
         for detail in details:
             # Consider anything not an exact match as a potential issue to review
             if not detail.get("is_exact_match", False):
                 field = detail.get("field", "unknown_field")
                 status = detail.get("status", "unknown_status")

                 if field not in mismatch_fields:
                     mismatch_fields[field] = {"count": 0, "examples": [], "statuses": {}}
                 mismatch_fields[field]["count"] += 1
                 mismatch_fields[field]["statuses"][status] = mismatch_fields[field]["statuses"].get(status, 0) + 1


                 # Store up to 3 examples for context
                 if len(mismatch_fields[field]["examples"]) < 3:
                     mismatch_fields[field]["examples"].append({
                         "case": result.get("case_name", "Unknown"),
                         "status": status,
                         "similarity": detail.get("similarity", "N/A"),
                         "ground_truth": detail.get("ground_truth"),
                         "extracted": detail.get("extracted"),
                         "processed_gt": detail.get("processed_ground_truth"),
                         "processed_ex": detail.get("processed_extracted")
                     })

    # Sort mismatches by frequency
    common_mismatches = sorted(
        [{"field": k, **v} for k, v in mismatch_fields.items()],
        key=lambda m: m["count"],
        reverse=True
    )[:20] # Show top 20 potential issues

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_cases_processed": len(results),
        "total_cases_valid": len(valid_results),
        "overall_metrics": {
            "accuracy": round(overall_accuracy, 4),
            "matches": total_matches,
            "partial_matches": total_partial_matches,
            "mismatches": total_mismatches,
            "missing_fields": total_missing, # Clarify meaning
            "total_fields_compared": total_fields
        },
        "case_metrics": case_metrics,
        "category_metrics": category_summary,
        "common_issues": common_mismatches # Renamed for clarity
    }


def generate_visualizations(summary: Dict, output_dir: str):
    """Generate visualizations from the summary report"""
    if not summary or "case_metrics" not in summary or not summary["case_metrics"]:
        print("Skipping visualizations: No valid case metrics found in summary.")
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.style.use('ggplot') # Use a clearer style

    # --- Case Accuracy Chart ---
    try:
        case_metrics_df = pd.DataFrame(summary["case_metrics"])
        if not case_metrics_df.empty and 'accuracy' in case_metrics_df.columns:
            case_metrics_df = case_metrics_df.sort_values("case_number") # Ensure sorted by number
            plt.figure(figsize=(max(6, len(case_metrics_df) * 0.5), 6)) # Dynamic width
            plt.bar(case_metrics_df["case_name"], case_metrics_df["accuracy"], color="#4CAF50") # Greenish
            plt.title("Accuracy per Case")
            plt.ylabel("Accuracy Score (Matches + Partials / Total)")
            plt.xlabel("Case Name")
            plt.xticks(rotation=90) # Rotate labels if many cases
            plt.ylim(0, 1.05) # Y-axis from 0 to 1
            plt.grid(axis='y', linestyle='--')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "case_accuracy.png"))
            plt.close()
        else:
            print("Skipping case accuracy chart: No data or 'accuracy' column missing.")
    except Exception as e:
        print(f"Error generating case accuracy chart: {e}")
        plt.close() # Ensure plot is closed on error

    # --- Category Accuracy Chart ---
    try:
        if summary.get("category_metrics"):
            category_data = [{"Category": k, **v} for k, v in summary["category_metrics"].items()]
            category_df = pd.DataFrame(category_data)
            if not category_df.empty and 'accuracy' in category_df.columns:
                category_df = category_df.sort_values("accuracy", ascending=False)
                plt.figure(figsize=(12, 7))
                plt.bar(category_df["Category"], category_df["accuracy"], color="#2196F3") # Blueish
                plt.title("Average Accuracy per Category")
                plt.ylabel("Accuracy Score")
                plt.xlabel("Data Category")
                plt.xticks(rotation=45, ha="right")
                plt.ylim(0, 1.05)
                plt.grid(axis='y', linestyle='--')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "category_accuracy.png"))
                plt.close()
            else:
                print("Skipping category accuracy chart: No data or 'accuracy' column missing.")
        else:
             print("Skipping category accuracy chart: No category metrics found.")
    except Exception as e:
        print(f"Error generating category accuracy chart: {e}")
        plt.close()

    # --- Common Issues Chart ---
    try:
        if summary.get("common_issues"):
            issue_df = pd.DataFrame(summary["common_issues"])
            if not issue_df.empty and 'count' in issue_df.columns:
                issue_df = issue_df.sort_values("count", ascending=True) # Barh looks better ascending
                plt.figure(figsize=(10, 8)) # Adjust size for horizontal bars
                plt.barh(issue_df["field"], issue_df["count"], color="#FF9800") # Orangish
                plt.title(f"Top {len(issue_df)} Fields with Issues (Mismatches, Partials, Missing)")
                plt.xlabel("Number of Issues Across Cases")
                plt.ylabel("Field Path")
                plt.grid(axis='x', linestyle='--')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "common_issues.png"))
                plt.close()
            else:
                print("Skipping common issues chart: No data or 'count' column missing.")
        else:
            print("Skipping common issues chart: No common issues found.")
    except Exception as e:
        print(f"Error generating common issues chart: {e}")
        plt.close()

def save_detailed_report(summary: Dict, results: List[Dict], output_dir: str):
    """Save a detailed report as JSON and HTML"""
    if not summary:
        print("Cannot save report: Summary data is missing.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Save summary as JSON
    try:
        summary_path = os.path.join(output_dir, "summary_report.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary JSON saved to {summary_path}")
    except Exception as e:
        print(f"Error saving summary JSON: {e}")

    # Generate and save HTML report
    try:
        html_report = generate_html_report(summary, results) # Pass results for potential detailed view
        html_path = os.path.join(output_dir, "detailed_report.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        print(f"HTML report saved to {html_path}")
    except Exception as e:
        print(f"Error generating or saving HTML report: {e}")


def generate_html_report(summary: Dict, results: List[Dict]) -> str:
    """Generate an HTML report from the summary and results"""
    # Helper to safely get nested keys
    def safe_get(data, keys, default="N/A"):
        val = data
        try:
            for key in keys:
                val = val[key]
            # Format numbers nicely
            if isinstance(val, float): return f"{val:.2%}" if key == 'accuracy' else f"{val:.2f}"
            if isinstance(val, int): return f"{val:,}" # Add commas to integers
            return val if val is not None else default
        except (KeyError, TypeError, IndexError):
            return default

    # Overall Metrics HTML
    om = summary.get('overall_metrics', {})
    # Get raw values first
    raw_matches = om.get('matches', 0)
    raw_partial = om.get('partial_matches', 0)
    raw_mismatches = om.get('mismatches', 0) # Mismatches value from summary
    raw_total = om.get('total_fields_compared', 0)

    # Calculate percentages safely
    match_pct_str = f"({raw_matches / raw_total:.1%})" if raw_total > 0 else "(0.0%)"
    partial_pct_str = f"({raw_partial / raw_total:.1%})" if raw_total > 0 else "(0.0%)"
    mismatch_pct_str = f"({raw_mismatches / raw_total:.1%})" if raw_total > 0 else "(0.0%)" # Use raw_mismatches

    overall_metrics_html = f"""
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Cases Processed</td><td>{safe_get(summary, ['total_cases_processed'])}</td></tr>
            <tr><td>Total Valid Cases</td><td>{safe_get(summary, ['total_cases_valid'])}</td></tr>
            <tr><td>Overall Accuracy</td><td class="{_get_accuracy_class(om.get('accuracy', 0))}">{safe_get(om, ['accuracy'])}</td></tr>
            <tr><td>Total Fields Compared</td><td>{safe_get(om, ['total_fields_compared'])}</td></tr>
            <tr><td>Exact Matches</td><td>{safe_get(om, ['matches'])} {match_pct_str}</td></tr>
            <tr><td>Partial Matches</td><td>{safe_get(om, ['partial_matches'])} {partial_pct_str}</td></tr>
            <tr><td>Mismatches (incl. missing)</td><td>{safe_get(om, ['mismatches'])} {mismatch_pct_str}</td></tr>
        </table>
    """

    # Case Metrics HTML
    case_metrics_html = """
        <table>
            <tr><th>Case</th><th>Accuracy</th><th>Exact</th><th>Partial</th><th>Mismatch</th><th>Total</th></tr>
    """
    for case in summary.get("case_metrics", []):
        acc = case.get('accuracy', 0)
        case_metrics_html += f"""
            <tr>
                <td>{safe_get(case, ['case_name'])}</td>
                <td class="{_get_accuracy_class(acc)}">{safe_get(case, ['accuracy'])}</td>
                <td>{safe_get(case, ['matches'])}</td>
                <td>{safe_get(case, ['partial_matches'])}</td>
                <td>{safe_get(case, ['mismatches'])}</td>
                <td>{safe_get(case, ['total'])}</td>
            </tr>
        """
    case_metrics_html += "</table>"

    # Category Metrics HTML
    category_metrics_html = """
        <table>
            <tr><th>Category</th><th>Accuracy</th><th>Exact</th><th>Partial</th><th>Mismatch</th><th>Total</th></tr>
    """
    sorted_categories = sorted(summary.get("category_metrics", {}).items(),
                              key=lambda item: item[1].get("accuracy", 0),
                              reverse=True)
    for category, metrics in sorted_categories:
        acc = metrics.get('accuracy', 0)
        category_metrics_html += f"""
            <tr>
                <td>{category}</td>
                <td class="{_get_accuracy_class(acc)}">{safe_get(metrics, ['accuracy'])}</td>
                <td>{safe_get(metrics, ['matches'])}</td>
                <td>{safe_get(metrics, ['partial_matches'])}</td>
                <td>{safe_get(metrics, ['mismatches'])}</td>
                <td>{safe_get(metrics, ['total'])}</td>
            </tr>
        """
    category_metrics_html += "</table>"

    # Common Issues HTML
    common_issues_html = ""
    if summary.get("common_issues"):
        common_issues_html = """
            <h2>Common Issues (Top Fields with Mismatches, Partials, Missing)</h2>
            <div class="chart"><img src="common_issues.png" alt="Common Issues Chart" style="max-width: 100%;"></div>
            <table>
                <tr><th>Field</th><th>Total Issues</th><th>Issue Types (Count)</th><th>Examples (Max 3)</th></tr>
        """
        for issue in summary["common_issues"]:
            statuses_html = "<br>".join([f"{k}: {v}" for k, v in issue.get('statuses', {}).items()])
            examples_html = ""
            for ex in issue.get("examples", []):
                 # Escape HTML characters in values to prevent rendering issues
                 gt_val = str(ex.get('ground_truth', 'N/A')).replace('&', '&').replace('<', '<').replace('>', '>')
                 ex_val = str(ex.get('extracted', 'N/A')).replace('&', '&').replace('<', '<').replace('>', '>')
                 examples_html += f"""
                     <div class='example'>
                         <strong>{safe_get(ex, ['case'])} ({safe_get(ex, ['status'])}, Sim: {safe_get(ex,['similarity'])})</strong><br>
                         GT: <code class='value'>{gt_val}</code><br>
                         EX: <code class='value'>{ex_val}</code>
                     </div>
                 """
            common_issues_html += f"""
                <tr>
                    <td><code>{safe_get(issue, ['field'])}</code></td>
                    <td>{safe_get(issue, ['count'])}</td>
                    <td>{statuses_html}</td>
                    <td>{examples_html}</td>
                </tr>
            """
        common_issues_html += "</table>"

    # Main HTML structure
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>JSON Comparison Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f4f7f6; color: #333; }}
            h1, h2, h3 {{ color: #0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 5px; }}
            h1 {{ font-size: 2em; }} h2 {{ font-size: 1.5em; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 25px; box-shadow: 0 2px 3px rgba(0,0,0,0.1); background-color: #fff; }}
            th, td {{ padding: 10px 12px; text-align: left; border: 1px solid #ddd; }}
            th {{ background-color: #e9ecef; color: #495057; font-weight: 600; }}
            tr:nth-child(even) {{ background-color: #f8f9fa; }}
            tr:hover {{ background-color: #e9ecef; }}
            .good {{ color: #28a745; font-weight: bold; }}
            .bad {{ color: #dc3545; font-weight: bold; }}
            .warning {{ color: #ffc107; font-weight: bold; }}
            .chart {{ margin: 30px auto; text-align: center; background-color: #fff; padding: 15px; border-radius: 5px; box-shadow: 0 2px 3px rgba(0,0,0,0.1); }}
            .chart img {{ max-width: 90%; height: auto; border: 1px solid #ddd; }}
            .example {{ border-left: 3px solid #007bff; padding-left: 10px; margin-bottom: 8px; font-size: 0.9em; }}
            .value {{ background-color: #eef; padding: 2px 4px; border-radius: 3px; font-family: monospace; word-break: break-all; }}
            code {{ background-color: #f1f1f1; padding: 2px 5px; border-radius: 3px; font-family: monospace; }}
        </style>
    </head>
    <body>
        <h1>JSON Comparison Report</h1>
        <p>Generated on: {safe_get(summary, ['timestamp'])}</p>

        <h2>Overall Metrics</h2>
        {overall_metrics_html}

        <div class="chart"><img src="case_accuracy.png" alt="Case Accuracy Chart"></div>
        <div class="chart"><img src="category_accuracy.png" alt="Category Accuracy Chart"></div>

        <h2>Case-by-Case Metrics</h2>
        {case_metrics_html}

        <h2>Category Metrics</h2>
        {category_metrics_html}

        {common_issues_html}

        <!-- Placeholder for future detailed results per case -->
        <!-- <h2>Detailed Field Comparison (Example Case)</h2> -->
        <!-- Add logic here to include detailed tables for each case if needed -->

    </body>
    </html>
    """
    return html


def _get_accuracy_class(accuracy):
    """Get CSS class based on accuracy value"""
    # Ensure accuracy is treated as a number
    try:
        acc_float = float(accuracy)
        if acc_float >= 0.9: return "good"
        if acc_float >= 0.7: return "warning"
        return "bad"
    except (ValueError, TypeError):
        return "" # Return no class if accuracy is not a number


def main(base_dir="synthetic_output", output_dir="comparison_results"):
    """Main function to process all cases and generate reports"""
    print(f"Starting comparison process...")
    print(f"Searching for case directories in: {os.path.abspath(base_dir)}")
    case_dirs = find_case_directories(base_dir)

    if not case_dirs:
        print(f"Error: No directories matching 'case_*' found in {base_dir}")
        return

    print(f"Found {len(case_dirs)} case directories.")

    results = []
    for i, case_dir in enumerate(case_dirs):
        print(f"\nProcessing Case {i+1}/{len(case_dirs)}: {os.path.basename(case_dir)}...")
        result = process_case(case_dir)
        if result:
            results.append(result)
            print(f"-> Completed: Accuracy = {result.get('accuracy', 0):.2%}")
        else:
            print(f"-> Failed or Skipped.")


    if not results:
        print("\nError: No cases could be processed successfully.")
        return

    print("\nGenerating summary report...")
    summary = generate_summary_report(results)

    if not summary or "error" in summary:
         print(f"Error generating summary: {summary.get('error', 'Unknown error')}")
         return

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {os.path.abspath(output_dir)}")
    except OSError as e:
        print(f"Error creating output directory {output_dir}: {e}")
        return

    print("Generating visualizations...")
    generate_visualizations(summary, output_dir)

    print("Saving detailed report...")
    save_detailed_report(summary, results, output_dir) # Pass results to save_detailed_report

    print("\nComparison process finished!")
    print(f"Overall Accuracy: {summary.get('overall_metrics', {}).get('accuracy', 0):.2%}")
    print(f"Reports saved to: {os.path.abspath(output_dir)}")

    return summary, results


if __name__ == "__main__":
    main() # Uses default "synthetic_output" and "comparison_results"