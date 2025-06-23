from setuptools import find_packages, setup

setup(
    name = 'options_pricer',
    version= '0.0.1',
    author= 'Sean Spencer',
    packages= find_packages(),
    install_requires=[
        "pandas",
        "pandas-market-calendars",
        "pyarrow",
        "fastparquet",
        "sqlalchemy",
        "requests",
        "python-dotenv",
        "boto3",
        "APScheduler",
    ],

)   