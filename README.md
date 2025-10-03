# Multifactor Safe Speed Estimation for Vehicles Using Monocular Camera Data

This project presents a dynamic speed limit estimation system that relies on monocular camera data to provide real-time, situation-specific safe speed recommendations for vehicles. The system is designed to improve vehicle safety by adapting to changing road conditions, especially where static speed limits are insufficient or non-existent.

---

##  Abstract

Traditional speed limits frequently struggle to adjust to real-time road conditions, potentially undermining vehicle safety.  
This study presents a dynamic speed limit estimation system that relies on monocular camera data. The method uses three techniques:

1. **Environment Perception** – Deep learning models classify environmental conditions (weather, visibility, time of day, and road type). These are processed by a regression model trained on statistically adjusted accident data to recommend safe speeds.  
2. **Road Geometry Analysis** – Evaluates curvature and road geometry to determine safe navigation speeds.  
3. **Vehicle Tracking** – Computes a safe speed based on the gap between the ego vehicle and the leading vehicle.  

For the final speed limit suggestion, the **lowest of the three estimates** is selected to prioritize safety.  

The system was tested using **software-in-the-loop (SIL)** evaluations with the **ApolloScape dataset** and **IPG Carmaker simulator**, showing adaptability in various driving scenarios. Results to be published in paper.

---

##  Key Features

- **Dynamic Speed Estimation**: Real-time safe driving speed calculation.  
- **Multi-Method Approach**: Combines three methodologies for robust recommendations.  
- **Environment Perception**: Deep learning classification of external conditions.  
- **Road Geometry Analysis**: Speed estimation for turns and curves.  
- **Vehicle Tracking**: Safe following distance and speed computation.  

---

### Overview
1. **Accident data based estimation**  
   <img src="Images/Picture 1.png" alt="System Architecture" width="400">

2. **Turning angle based estimation**  
   <img src="Images/Picture 2.png" alt="Road Segmentation" width="400">

3. **Leading vehicle based estimation**  
   <img src="Images/Picture 3.png" alt="Object Detection" width="400">  

---

### Output Videos
Real-time demonstrations of the safe speed estimation system in action:

1. **City Driving Scenario** – Estimating safe speeds in an urban environment.  
   ![City Driving Demo](integrated_safe_speed_output.gif)

2. **Sharp Turn Navigation** – Adjusting speeds for sharp turns using road geometry analysis.  
   ![Sharp Turn Demo](integrated_safe_speed_output-2.gif)

---

## Example Usage

### Test Videos Available
Sample test videos are provided in the `testVideos/` folder for different driving scenarios:
- `straight_1.avi`, `straight_2.avi` - Straight road driving
- `turning_1.avi`, `turning_2.avi`, `turning_3.avi` - Various turning scenarios
- `slightCurve.avi` - Slight curve navigation
- `roundAbout.avi` - Roundabout navigation

### Running the System

#### Option 1: Run All Methods (Integrated System)
To run the complete integrated safe speed estimation system with all three methods:

```bash
python EstimationMethods/integrated_safe_speed_system.py
```

Update the video file path in the script to point to your desired test video or custom video file.

#### Option 2: Run Individual Methods
To run individual speed estimation methods separately:

**Environment-based Speed Estimation:**
```bash
python EstimationMethods/RegressionMethodVideo.py
```

**Road Curvature-based Speed Estimation:**
```bash
python EstimationMethods/roadCurvatureDetectionLinesVideoV3.py
```

**Vehicle Tracking-based Speed Estimation:**
```bash
python EstimationMethods/yolo_simple_speed_v3.py
```

### Custom Video Input
For custom videos, update the input video path in the respective Python scripts before running.

---

## Additional Resources

### IPG CarMaker-MATLAB Setup
Software-in-the-loop (SIL) simulation files and MATLAB/Simulink simulation setup is available in the `IPG Carmaker-MATLAB Setup/` folder.

### Scenario Output Results
Pre-generated results for various driving scenarios are stored in the `scenarioOutputs/` folder:
- `straight/` - Straight road driving results
- `intersection/` - Intersection navigation results  
- `roundAbout/` - Roundabout scenario results
- `SlightTurn/` - Slight turn navigation results

Each scenario folder contains outputs from different estimation methods:
- `turnAngleBasedOutput/` - Road geometry analysis results
- `stoppingDistanceBased/` - Vehicle tracking analysis results  
- `DataBasedOutput/` - Environment perception analysis results

Special thanks to:
- [@aerostar24](https://github.com/aerostar24) for project supervision and review
