import os
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

import subprocess
import sys

subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/main.py"], env=os.environ.copy())