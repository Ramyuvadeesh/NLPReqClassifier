import os
import random

def generate_synthetic_113_arff():
    output_dir = "dataset"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "synthetic_test_113.arff")
    
    print(f"Generating {file_path} with exactly 113 requirements...")
    
    # Class distribution scaled to exactly 113 instances based on Table 5 proportions
    distribution = {
        "F": 25, "US": 18, "SE": 18, "O": 16, "A": 11, 
        "LF": 6, "PE": 5, "MN": 5, "SC": 4, "L": 3, "FT": 2
    }
    
    templates = {
        "F": [
            "The system shall display the search results in a vertical table.",
            "The product shall record all the equipment that has been reserved.",
            "The system shall allow the user to view existing dispute cases.",
            "The application must allow a user to identify himself or herself as a player.",
            "The system shall generate an audit report based on the available data."
        ],
        "US": [
            "The user interface shall be intuitive and easy to learn.",
            "90% of new users shall be able to start the display within 90 minutes.",
            "The system shall require no formal training for standard data entry tasks.",
            "The application must provide clear error messages to avoid making mistakes."
        ],
        "SE": [
            "The system must encrypt all user passwords in the database.",
            "Only authenticated users with administrator privileges shall have access.",
            "The application shall protect against unauthorized access and malicious attacks.",
            "The product shall ensure that it can only be accessed by authorized users."
        ],
        "O": [
            "The product shall operate seamlessly within the Windows XP operating system.",
            "The web application must be fully compatible with HTML browsers.",
            "The system shall interface directly with the existing central server.",
            "The software will function alongside server software on any operating system."
        ],
        "A": [
            "The product shall be available 24 hours a day, 7 days a week.",
            "The system shall achieve 99.9% up time during normal business hours.",
            "Scheduled maintenance outages will not exceed the scheduled maintenance windows."
        ],
        "LF": [
            "The application shall conform to the corporate branding and color scheme.",
            "The user interface must maintain a professional appearance and standard layout.",
            "All screens shall comply with corporate standards for interface creation."
        ],
        "PE": [
            "The application must load the search results in under 3 seconds.",
            "The maximum response time for any database query shall be no more than 5 seconds.",
            "The system shall render the graphics fast to keep up-to-date data in the display."
        ],
        "MN": [
            "The code must be highly configurable for use with various database management systems.",
            "New software updates shall be deployable within 10 minutes.",
            "The system shall allow administrators to easily integrate new map data."
        ],
        "SC": [
            "The web server must be capable of supporting 5,000 simultaneous users.",
            "The system shall handle up to 10,000 concurrent database connections.",
            "The architecture must be able to support multiple remote users."
        ],
        "L": [
            "The product must comply with Sarbanes-Oxley auditing regulations.",
            "The system shall conform to the legal requirements as specified by Regulation E."
        ],
        "FT": [
            "The system shall automatically restore user preferences in the event of a failure.",
            "The application must operate in offline mode whenever internet connection is unavailable."
        ]
    }

    with open(file_path, "w", encoding="utf-8") as f:
        # Write ARFF Headers
        f.write("@RELATION synthetic_nfr_113\n\n")
        f.write("@ATTRIBUTE ProjectID numeric\n")
        f.write("@ATTRIBUTE RequirementText string\n")
        f.write("@ATTRIBUTE class {F,A,L,LF,MN,O,PE,SC,SE,US,FT}\n\n")
        f.write("@DATA\n")
        
        project_id = 200
        for req_class, count in distribution.items():
            for _ in range(count):
                text = random.choice(templates[req_class])
                f.write(f"{project_id},'{text}',{req_class}\n")
                project_id += 1

    print("Successfully generated 113 synthetic requirements for hold-out testing.")

if __name__ == "__main__":
    generate_synthetic_113_arff()