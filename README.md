# Project Manzar 🌍
#### Video Demo:  https://youtu.be/YCjhownqwUc

**Advanced Satellite Imagery Change Detection System**

Project Manzar is a comprehensive satellite imagery analysis tool that detects and classifies changes in land use over time using Google Earth Engine, machine learning algorithms, and advanced computer vision techniques.

## 🚀 Features

### Core Functionality
- **Multi-temporal Analysis**: Compare satellite images from different time periods (2020 vs 2025)
- **Advanced Change Detection**: Multiple detection methods including traditional pixel differencing and ML-based approaches
- **Spectral Analysis**: Calculate NDVI, NDBI, and MNDWI indices for detailed land classification
- **Change Classification**: Categorize changes into development, degradation, water changes, and mixed changes
- **Water Filtering**: Intelligent water area detection to reduce false positives from water color variations

### Detection Methods
1. **Traditional Change Detection**
   - Pixel-wise difference analysis
   - Adaptive thresholding with Otsu's method
   - Morphological operations for noise reduction
   - Water area filtering

2. **Machine Learning Approach**
   - PCA-based feature extraction from 5x5 image blocks
   - K-means clustering for change region identification
   - Advanced noise reduction and region filtering

3. **Spectral Analysis**
   - NDVI (Normalized Difference Vegetation Index)
   - NDBI (Normalized Difference Built-up Index) 
   - MNDWI (Modified Normalized Difference Water Index)

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- Google Earth Engine account
- Google Maps API key

### Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Project-Manzar
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Google Earth Engine Authentication**
   ```bash
   earthengine authenticate
   ```

5. **Configure API Key**
   - Replace `YOUR_GOOGLE_MAPS_API_KEY` in `main.py` with your actual Google Maps API key
   - Get your API key from [Google Cloud Console](https://console.cloud.google.com/)

## 📖 Usage

### Basic Usage
```bash
python main.py
```

The script will prompt you to:
1. Enter a location name (e.g., "Dubai", "New York", "Tokyo")
2. Select from suggested locations
3. Automatically process the area and generate results

### Example
```bash
$ python main.py
📍 Enter a location to search for: Dubai

🔍 Found potential locations:
  1. Dubai - United Arab Emirates
  2. Dubai Creek - Dubai - United Arab Emirates
  3. Dubai Marina - Dubai - United Arab Emirates
  ...

👉 Pick a location (1-5): 1

🛰️  Processing location: Dubai - United Arab Emirates (25.2048, 55.2708)
```

## 📊 Output Files

The system generates comprehensive analysis results in the `output/` directory:

### Core Images
- `image_before.png` - Satellite image from 2020 with timestamp
- `image_after.png` - Satellite image from 2025 with timestamp
- `ChangeMap.jpg` - Traditional change detection result (water-filtered)
- `difference.jpg` - Raw pixel difference image
- `difference_filtered.jpg` - Water-filtered difference image

### Machine Learning Results
- `ml_difference.jpg` - ML method difference image
- `ml_change_map.jpg` - Raw ML change detection result
- `ml_clean_change_map.jpg` - Final ML result with noise reduction

### Analysis Files
- `water_mask.jpg` - Detected water areas (filtered out)
- `change_classification.png` - Color-coded change classification:
  - 🟢 **Green**: Sand/Grass → Buildings (Development)
  - 🔴 **Red**: Buildings → Sand/Grass (Abandonment)
  - 🔵 **Blue**: Water Changes (Ignored)
  - 🟡 **Yellow**: Mixed Changes (Ignored)
  - ⚫ **Black**: No Meaningful Change

## 🔬 Technical Details

### Data Sources
- **Satellite Imagery**: Copernicus Sentinel-2 (COPERNICUS/S2_SR_HARMONIZED)
- **Cloud Filtering**: <10% cloud coverage
- **Resolution**: 15m per pixel
- **Spectral Bands**: RGB (B4, B3, B2) for visualization, full spectrum for analysis

### Algorithms
- **Change Detection**: Multi-threshold approach with Otsu's method
- **ML Pipeline**: PCA + K-means clustering on 5x5 feature blocks
- **Morphological Operations**: Opening, closing, and connected component analysis
- **Water Detection**: HSV color space analysis with multiple criteria

### Performance Optimizations
- Automatic image resizing for ML processing
- Efficient memory management for large satellite images
- Parallel processing where possible
- Smart region of interest (ROI) selection

## 🎯 Use Cases

- **Urban Development Monitoring**: Track city expansion and infrastructure growth
- **Environmental Assessment**: Monitor deforestation, desertification, and land degradation
- **Disaster Impact Analysis**: Assess damage from natural disasters
- **Agricultural Monitoring**: Track crop changes and irrigation patterns
- **Research Applications**: Academic studies on land use change

## 📈 Sample Results

The system provides detailed statistics:
```
📊 Change Analysis Summary:
   🟢 Sand→Buildings Development: 15.5%
   🔴 Buildings→Sand Abandonment: 8.2%
   🔵 Water Changes: 2.1% (ignored)
   🟡 Mixed Changes: 12.3% (ignored)
   ⚫ No Meaningful Change: 61.9%
```

## 🔧 Configuration

### Key Parameters (in `main.py`)
- `lon_offset`: Width of analysis area (~8.8 km)
- `lat_offset`: Height of analysis area (~5 km)
- `min_area`: Minimum change region size for filtering
- `cloud_threshold`: Maximum cloud coverage percentage

### Customization
- Modify spectral indices calculations
- Adjust change detection thresholds
- Add new classification categories
- Implement additional ML algorithms

## 🐛 Troubleshooting

### Common Issues
1. **Google Earth Engine Authentication Error**
   ```bash
   earthengine authenticate
   ```

2. **API Key Issues**
   - Ensure your Google Maps API key is valid and has geocoding enabled
   - Check API quotas and billing

3. **Memory Issues with Large Images**
   - The system automatically resizes images for processing
   - Reduce ROI size if needed

4. **No Clear Images Found**
   - Try different years or locations
   - Check cloud coverage in the area

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Earth Engine** for satellite imagery access
- **Google Maps API** for geocoding services
- **OpenCV** and **scikit-learn** for image processing and ML algorithms
- **Sentinel-2** satellite data from Copernicus program

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the development team
- Check the documentation for common solutions

---

**Project Manzar** - *Transforming satellite imagery into actionable insights* 🌍✨
