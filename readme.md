# Gait Analysis and Person Identification System

### Git URL

```bash
git clone https://github.com/akash-banger/semusi_assignment.git
```

## Setup

### Prerequisites

create a new virtual environment

```bash
python -m venv gait_env
```

activate the virtual environment

```bash
source gait_env/bin/activate
```

install the dependencies

```bash
pip install -r requirements.txt
```

### Database Configuration

The system requires a PostgreSQL database with the following tables:
- `persons`: Stores basic person information
- `gait_features`: Stores extracted gait features
- `raw_gait_data`: Stores raw sensor data

Set your database connection string in the environment:


```bash
export DB_CONNECTION_STRING="postgresql://user:password@localhost:5432/gait_db"
```



## Usage

### 1. Store Profiles
```bash
python src/store_profiles.py
```

Store profiles for all people in the training dataset

```bash
python src/store_profiles.py
```



### 2. Identify Person

```bash
python src/identify_person.py
```






## Results

![Results](./results.png)