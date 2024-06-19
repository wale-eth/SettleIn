## SettleIn: AI-Powered Location Recommender
SettleIn is an AI-driven tool designed to help users find the perfect neighborhood in 10 major English cities. By analyzing user preferences and considering factors such as cost of living, safety, and local amenities, SettleIn provides personalized recommendations for the ideal place to call home.


### Features

Utilizes advanced Multi-Layer Perceptron (MLP) algorithms for accurate recommendations
Considers a wide range of factors, including cost of living, safety, and local amenities
Provides personalized suggestions tailored to user preferences
Offers recommendations for neighborhoods in 10 major English cities
User-friendly interface for seamless interaction

### Installation

Clone the repository:
Copygit clone https://github.com/wale-eth/SettleIn.git

Install the required dependencies:
Copycd SettleIn
pip install -r requirements.txt


### Usage

Prepare your input data in the required format (e.g., CSV file) and place it in the data directory.
Run the main script to train the models and generate recommendations:
Copypython settle_in.py --input_data data/your_data.csv --output_dir results/

The script will read the input data, train various machine learning models, and evaluate their performance. The best-performing model (MLP) will be used to generate personalized recommendations.
The generated recommendations will be saved in the specified output directory.

### Data
The input data should include the following attributes for each neighborhood:

- Cost of living index
- Crime Rate
- Rent index
- IMD Score
- Ethnic Distribution
- Unemployment Rate
...

Ensure that the input data is properly formatted and cleaned before running the script.

### Models
SettleIn trains and evaluates the following machine learning models:

- K-Nearest Neighbors (KNN)
- Multi-Layer Perceptron (MLP)
- Matrix Factorization (NMF)
- Singular Value Decomposition (SVD)
- Collaborative Filtering (CF)
- Random Forest (RF)
- Gradient Boosting Machine (GBM)
- XGBoost

The best-performing model (MLP) is selected based on evaluation metrics such as Mean Absolute Error (MAE) and Root Mean Square Error (RMSE).

### Results
The script generates the following outputs:

Trained models for each algorithm
Evaluation metrics for each model
Personalized recommendations for user preferences

The results are saved in the specified output directory.

### Contributing
Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

### License
This project is licensed under the MIT License.

### Acknowledgements
We would like to thank the following organizations for their valuable data and resources:

Numbeo: Cost of living, crime, and rent index data
Office for National Statistics: International migration data for England and Wales
