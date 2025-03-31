#!/usr/bin/env python3
"""
Direct test of RAG functionality with a specific SEC filing section
This isolates the RAG processing to verify it's working properly
"""

import os
import sys
import time
import json
import tempfile
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import the SEC RAG processor directly
sys.path.append('.')
try:
    from tools.rag_tool import sec_rag_processor
    logger.info("Successfully imported sec_rag_processor")
except ImportError as e:
    logger.error(f"Failed to import sec_rag_processor: {e}")
    sys.exit(1)

def create_test_content():
    """Create a sample content file to test RAG processing"""
    # This is a simplified version of a business section from a 10-K
    content = """
    Item 1. Business
    
    Company Overview
    
    Apple Inc. is a global technology company that designs, manufactures, and sells smartphones, personal computers, tablets, wearables, and accessories, as well as providing various related services. The Company's products include iPhone, Mac, iPad, AirPods, Apple TV, Apple Watch, HomePod, and various accessories. The Company also offers services such as AppleCare, iCloud, Apple Pay, Apple Card, Apple Music, Apple Fitness+, Apple News+, Apple Arcade, Apple TV+, and other services.
    
    The Company sells its products and services through its retail stores, online stores, and direct sales force, as well as through third-party cellular network carriers, wholesalers, retailers, and resellers. The Company's customers include consumers, small and mid-sized businesses, education, enterprise, and government customers.
    
    Products
    
    iPhone
    iPhone is the Company's line of smartphones based on its iOS operating system. iPhone includes various models with different features, storage capacities, and price points.
    
    Mac
    Mac is the Company's line of personal computers based on its macOS operating system. Mac includes laptops (MacBook Air and MacBook Pro) and desktops (iMac, Mac mini, Mac Studio, and Mac Pro).
    
    iPad
    iPad is the Company's line of multipurpose tablets based on its iPadOS operating system. iPad includes iPad Pro, iPad Air, iPad, and iPad mini.
    
    Wearables, Home, and Accessories
    This category includes products such as AirPods, Apple TV, Apple Watch, Beats products, HomePod, iPod touch, and various Apple-branded and third-party accessories.
    
    Services
    
    Digital Content Stores and Streaming Services
    The Company operates various platforms including the App Store, Apple Music, Apple TV+, Apple Arcade, Apple News+, and Apple Fitness+, which allow customers to discover and download applications and digital content.
    
    AppleCare
    The Company offers a portfolio of fee-based service and support products under the AppleCare brand, which provide customers with various support options.
    
    Cloud Services
    The Company's cloud services store and keep customers' content up-to-date and available across multiple Apple devices and Windows personal computers.
    
    Other Services
    The Company offers various other services, including Apple Card, Apple Pay, and Apple Cash, which are cash and payment services.
    
    Competition
    
    The markets for the Company's products and services are highly competitive and characterized by rapid technological advances. Many of the Company's competitors have significant resources and may be able to provide products and services at little or no profit or at a loss to compete with the Company's offerings.
    
    The smartphone, tablet, and personal computer markets are highly competitive and are characterized by aggressive price competition, frequent product introductions, and evolving industry standards. The Company's ability to compete successfully depends on its ability to ensure a continuing and timely introduction of innovative new products, services, and technologies to the marketplace.
    
    In the smartphone market, the Company competes with various manufacturers that sell devices based on the Android operating system, as well as other manufacturers. In the tablet market, the Company competes with various manufacturers that sell tablets based on the Android operating system, as well as Microsoft Corporation, which sells tablets based on the Windows operating system. In the personal computer market, the Company competes primarily with manufacturers that sell computers based on the Windows operating system.
    
    Supply Chain and Manufacturing
    
    The Company's products are manufactured primarily by outsourcing partners, mainly located in Asia. The Company also manufactures certain products in the United States and Ireland. The Company's outsourcing partners manufacture the Company's products using components obtained from suppliers, some of which are exclusive suppliers. Although certain components are currently obtained from single or limited sources, the Company works closely with its suppliers to manage supply and mitigate supply risks.
    
    Research and Development
    
    The Company believes that focused investments in research and development are critical to its future growth and competitive position in the marketplace. The Company's research and development spending is focused on developing new and improved products and services, as well as advancing technologies to help deliver innovative products and services.
    
    Intellectual Property
    
    The Company believes that its intellectual property is critical to its success and competitive position. The Company owns and licenses various patents, trademarks, copyrights, trade secrets, and other intellectual property rights. The Company also enters into confidentiality and invention assignment agreements with its employees and contractors, and maintains comprehensive policies regarding the handling of intellectual property to protect its rights.
    
    Human Capital
    
    The Company believes that its employees are vital to its success. The Company's total number of employees has grown over time, reflecting the Company's growth and expansion into new products, services, and markets. The Company is committed to hiring, developing, and retaining talented employees, and providing competitive compensation and benefits.
    
    Environmental, Social, and Governance (ESG)
    
    The Company is committed to various environmental, social, and governance initiatives. The Company aims to achieve carbon neutrality for its entire carbon footprint by certain target dates, including its manufacturing supply chain and product life cycle. The Company also focuses on various social initiatives, including diversity and inclusion, privacy and security, and supplier responsibility.
    
    Seasonality
    
    The Company has historically experienced higher net sales in its first quarter compared to other quarters in its fiscal year due to seasonal holiday demand. The Company recognizes the importance of this seasonal demand in its business planning.
    """
    
    # Create a temporary file
    fd, temp_path = tempfile.mkstemp(prefix="test_sec_section_", suffix=".txt")
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    
    logger.info(f"Created test content file at {temp_path} ({len(content)} characters)")
    return temp_path, len(content)

def test_rag_processing():
    """Test RAG processing directly with a sample content file"""
    # Create the test content
    temp_file_path, content_length = create_test_content()
    
    # Define test questions
    test_questions = [
        "What products does the company sell?",
        "Who are Apple's competitors?",
        "How does Apple manufacture its products?",
        "What is Apple's approach to intellectual property?",
        "Summarize Apple's business model"
    ]
    
    results = {}
    
    try:
        for question in test_questions:
            logger.info(f"Testing RAG with question: '{question}'")
            
            # Record start time
            start_time = time.time()
            
            # Process with RAG
            rag_result = sec_rag_processor(
                query=question, 
                file_path=temp_file_path
            )
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            
            # Log results
            processed_length = len(rag_result.get("data", ""))
            logger.info(f"RAG processing completed in {elapsed_time:.2f} seconds")
            logger.info(f"Original length: {content_length}, Processed length: {processed_length}")
            logger.info(f"Compression ratio: {processed_length/content_length:.2%}")
            
            # Store results
            results[question] = {
                "original_length": content_length,
                "processed_length": processed_length,
                "compression_ratio": processed_length/content_length,
                "processing_time": elapsed_time,
                "status": rag_result.get("status"),
                "answer": rag_result.get("data", "")
            }
            
            # Add a delay between questions
            time.sleep(2)
    
    finally:
        # Clean up the temp file
        try:
            os.remove(temp_file_path)
            logger.info(f"Removed temporary file: {temp_file_path}")
        except Exception as e:
            logger.error(f"Error removing temporary file: {e}")
    
    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"rag_direct_test_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "test_content_length": content_length,
            "results": results
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    # Print summary
    print("\nRAG PROCESSING TEST SUMMARY")
    print("=" * 50)
    print(f"Total questions tested: {len(test_questions)}")
    print(f"Original content length: {content_length} characters")
    
    avg_processed_length = sum(r["processed_length"] for r in results.values()) / len(results)
    avg_compression = avg_processed_length / content_length
    avg_time = sum(r["processing_time"] for r in results.values()) / len(results)
    
    print(f"Average processed length: {avg_processed_length:.0f} characters")
    print(f"Average compression ratio: {avg_compression:.2%}")
    print(f"Average processing time: {avg_time:.2f} seconds")
    print("\nSample Answers:")
    
    for question, result in results.items():
        print(f"\nQ: {question}")
        # Print just the first 150 chars of the answer
        answer = result["answer"]
        if len(answer) > 150:
            print(f"A: {answer[:150]}...")
        else:
            print(f"A: {answer}")

if __name__ == "__main__":
    print(f"Starting direct RAG test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    test_rag_processing() 