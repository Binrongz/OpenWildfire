import json
import os
import glob
from jinja2 import Environment, FileSystemLoader

def batch_generate_reports():
    """Generate HTML reports in bulk"""
    
    # Gets the directory where the script is currently located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define reports folder path
    reports_dir = os.path.join(script_dir, '..')
    
    # Check reports the folder exists
    if not os.path.exists(reports_dir):
        print(f"Error: reports Folder does not exist. Please check the path: {reports_dir}")
        return
    
    # Check all JSON files
    json_pattern = os.path.join(reports_dir, '*.json')
    json_files = glob.glob(json_pattern)
    
    if not json_files:
        print(f"Warrning: No JOSN files found in {reports_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # Set Jinja2 template directory to the script directory
    env = Environment(loader=FileSystemLoader(script_dir))
    
    # Loading the template
    try:
        template = env.get_template('report_template.html')
    except Exception as e:
        print(f"Error: Can not found HTML files. Please check the path: {os.path.join(script_dir, 'report_template.html')}")
        print(f"Detail error: {e}")
        return
    
    # Set the output directory：final_reports (reports/final_reports/)
    final_reports_dir = os.path.join(reports_dir, 'final_reports')
    os.makedirs(final_reports_dir, exist_ok=True)
    
    # Statistical information
    success_count = 0
    error_count = 0
    
    # Iterate over each JSON file
    for json_file_path in json_files:
        try:
            print(f"\nProcessing: {os.path.basename(json_file_path)}")
            
            # Loading JSON data
            with open(json_file_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            # Rendering HTML content
            output_html = template.render(report=report_data)
            
            # Build the output filename
            # Option 1: Use report id (if present)
            if 'report_header' in report_data and 'report_id' in report_data['report_header']:
                output_file_name = f"report_{report_data['report_header']['report_id']}.html"
            else:
                # Option 2: Use the original JSON filename
                json_basename = os.path.splitext(os.path.basename(json_file_path))[0]
                output_file_name = f"report_{json_basename}.html"
            
            output_file_path = os.path.join(final_reports_dir, output_file_name)
            
            # Writing to an HTML file
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(output_html)
            
            print(f"✓ Success: {output_file_name}")
            success_count += 1
            
        except json.JSONDecodeError as e:
            print(f"✗ JSON parse ({os.path.basename(json_file_path)}): {e}")
            error_count += 1
        except KeyError as e:
            print(f"✗ Data structure error ({os.path.basename(json_file_path)}): Missing fields {e}")
            error_count += 1
        except Exception as e:
            print(f"✗ Processing error ({os.path.basename(json_file_path)}): {e}")
            error_count += 1
    
    # Output statistic information
    print(f"\n" + "="*50)
    print(f"Batch processing done!")
    print(f"Success: {success_count} files")
    print(f"Fail: {error_count} files")
    print(f"Output directory: {final_reports_dir}")

if __name__ == "__main__":
    batch_generate_reports()