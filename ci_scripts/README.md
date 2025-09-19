### Setup ###
Python parser requires matlotlib module:
###
python -m venv venv
source ./venv/bin/activate
pip install matplotlib
###
### Usage ###
To run testing for any number of commits and parse the results to PDF report, use:
./ci_run_and_parse.sh <artifact_dir> <commits_list>
###
To parse the results of previously ran testing to PDF report, use:
./ci_check_diffs.sh <list_of_testing_directories> | python parse_to_pdf.py -o <output_path>
###
Or to get the parsed results in text format simply use:
./ci_check_diffs.sh <list_of_testing_directories>
###
You can also run testing for single commit without parsing by using:
./ci_runner <commit_id> <artifact_dir> <docker_image>

