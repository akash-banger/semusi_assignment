# Gait Analysis and Person Identification System

### Git URL

```bash
git clone https://github.com/akash-banger/semusi_assignment.git
```

A sophisticated biometric system that identifies individuals based on their unique walking patterns (gait) using sensor data from multiple body locations.

## Overview

This system uses machine learning and signal processing techniques to analyze gait patterns from various sensors (accelerometers and EMG) placed on different body locations. It can:
- Extract comprehensive gait features from sensor data
- Store individual gait profiles in a database
- Identify people based on their walking patterns

## System Architecture

### Components

1. **Gait Analyzer (`GaitAnalyzer` class)**
   - Feature extraction from sensor data
   - Profile management
   - Person identification
   - Database interactions

2. **Data Collection Points**
   - Right/Left Foot accelerometers
   - Right/Left Shin accelerometers
   - Right/Left Thigh accelerometers
   - Right/Left EMG sensors

### Features Extracted

- Magnitude-based features (mean, standard deviation, max, min)
- Statistical features (kurtosis, skewness)
- Gait characteristics (step variability, symmetry index)
- EMG measurements and symmetry

## Setup

### Prerequisites

create a new virtual environment

```bash
python -m venv gait_env
```

install the dependencies

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