from typing import Dict

def gfft_template() -> Dict:
    """Template JSON for the GFFT schema"""
    return {
        "personal_details": {
            "client": {
                "title": None,
                "first_name": None,
                "middle_names": None,
                "last_name": None,
                "known_as": None,
                "pronouns": None,
                "date_of_birth": None,
                "place_of_birth": None,
                "nationality": None,
                "gender": None,
                "legal_sex": None,
                "marital_status": None,
                "home_phone": None,
                "mobile_phone": None,
                "email_address": None,
            },
            "current_address": {
                "ownership_status": None,
                "postcode": None,
                "house_name_or_number": None,
                "street_name": None,
                "address_line3": None,
                "address_line4": None,
                "town_city": None,
                "county": None,
                "country": None,
                "move_in_date": None,
                "previous_addresses": [], 
            },
            "dependants_children": [],
        },
        "employment": {
            "client": {
                "country_domiciled": None,
                "resident_for_tax": None,
                "national_insurance_number": None,
                "employment_status": None,
                "desired_retirement_age": None,
                "occupation": None,
                "employer": None,
                "employment_started": None,
                "highest_rate_of_tax_paid": None,
                "notes": None,
            },
        },
        "incomes": {
            "owner": None,
            "name": None,
            "amount": None,
            "frequency": None,
            "net_gross": None,
            "timeframe": None
        },
        "expenses": {
            "loan_repayments": {
                "owner": None,
                "name": None,
                "amount": None,
                "frequency": None,
                "priority": None,
                "timeframe": None
            },
            "housing_expenses": {
                "owner": None,
                "name": None,
                "amount": None,
                "frequency": None,
                "priority": None,
                "timeframe": None
            },
            "motoring_expenses": {
                "owner": None,
                "name": None,
                "amount": None,
                "frequency": None,
                "priority": None,
                "timeframe": None
            },
            "personal_expenses": {
                "owner": None,
                "name": None,
                "amount": None,
                "frequency": None,
                "priority": None,
                "timeframe": None
            },
            "professional_expenses": {
                "owner": None,
                "name": None,
                "amount": None,
                "frequency": None,
                "priority": None,
                "timeframe": None
            },
            "miscellaneous_expenses": {
                "owner": None,
                "name": None,
                "amount": None,
                "frequency": None,
                "priority": None,
                "timeframe": None
            },
            "notes": None,
        },
        "pensions": {
            "owner": None,
            "type": None,
            "provider": None,
            "value": None,
            "policy_number": None
        },
        "savings_investments": {
            "owner": None,
            "type": None,
            "provider": None,
            "value": None
        },
        "other_assets": [],
        "loans_mortgages": {
            "owner": None,
            "type": None,
            "provider": None,
            "monthly_cost": None,
            "outstanding_value": None,
            "interest_rate": None,
            "special_rate": None,
            "final_payment": None
        },
        "health_details": {
            "client": {
                "current_state_of_health": None,
                "state_of_health_explanation": None,
                "smoker": None,
                "cigarettes_per_day": None,
                "smoker_since": None,
                "long_term_care_needed": None,
                "long_term_care_explanation": None,
                "will": None,
                "information_about_will": None,
                "power_of_attorney": None,
                "attorney_details": None,
            },
        },
        "protection_policies": {
            "owner": None,
            "type": None,
            "provider": None,
            "monthly_cost": None,
            "amount_assured": None,
            "in_trust": None,
            "assured": None   
        },
        "objectives": None,
    } 