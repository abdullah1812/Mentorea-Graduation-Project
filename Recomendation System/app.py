from flask import Flask, request, jsonify
import pickle
import numpy as np
from lightfm import LightFM
from scipy.sparse import coo_matrix, csr_matrix, hstack
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine, text
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database connection string
DATABASE_URL = "mssql+pyodbc://db21354:aF-7K6x%214k%23E@db21354.public.databaseasp.net/db21354?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=no&MultipleActiveResultSets=True"

app = Flask(__name__)

# Helper function to check for non-finite values
def check_finite(matrix, name):
    if isinstance(matrix, (coo_matrix, csr_matrix)):
        data = matrix.data
    else:
        data = matrix
    if np.any(~np.isfinite(data)):
        n_nans = np.isnan(data).sum()
        n_infs = np.isinf(data).sum()
        logger.error(f"{name} contains {n_nans} NaNs and {n_infs} infinite values")
        raise ValueError(f"Non-finite values found in {name}")
    logger.info(f"{name} is finite")

# Fetch data from database
def fetch_data_from_database():
    engine = create_engine(DATABASE_URL)
    try:
        with engine.connect() as connection:
            logger.info("Database connection successful")

           
            query1 = text("""
                SELECT Id, Name, Location, About FROM AspNetUsers
                WHERE Rate IS NULL
                AND PriceOfSession = 0
                AND NumberOfSession IS NULL
                AND NumberOfExperience IS NULL
            """)
            df = pd.read_sql(query1, connection)

            # Query MenteeFields
            query2 = text("SELECT * FROM MenteeFields")
            df2 = pd.read_sql(query2, connection)
            df2.rename(columns={'MenteeId': 'Id'}, inplace=True)

            result1 = pd.merge(df, df2, on='Id', how='outer')

            # Query Fields
            query3 = text("SELECT * FROM Fields")
            df3 = pd.read_sql(query3, connection)
            df3.rename(columns={'Id': "FieldId"}, inplace=True)

            result2 = pd.merge(result1, df3, on='FieldId', how='outer')

            # Query Specializations
            query4 = text("SELECT * FROM Specializations")
            df4 = pd.read_sql(query4, connection)
            df4.rename(columns={'Id': "SpecializationId", 'Name': "SpecializationName"}, inplace=True)

            menteefinalresult = pd.merge(result2, df4, on='SpecializationId', how='outer')
            menteefinalresult.drop(columns=["FieldId", "SpecializationId"], inplace=True)

            # Aggregate and impute missing values
            menteefinalresult = menteefinalresult.groupby('Id').agg({
                'Name': 'first',
                'Location': 'first',
                'About': 'first',
                'FieldName': lambda x: ' '.join(x.dropna()),
                'SpecializationName': 'first'
            }).reset_index()

            menteefinalresult.dropna(inplace=True)

            if menteefinalresult.empty:
                logger.error("No mentees found matching the criteria.")
                raise ValueError("Mentee data is empty.")
            menteefinalresult.to_csv("final_mentees_db.csv", index=False)

            # Query Mentors
            query = text("""
                SELECT Id, Name, Location, About, FieldId, Rate, NumberOfExperience FROM AspNetUsers
                WHERE
                NumberOfSession IS NOT NULL
                AND NumberOfExperience IS NOT NULL
            """)
            df01 = pd.read_sql(query, connection)
            result2 = pd.merge(df01, df3, on='FieldId', how='outer')
            mentor_finalresult = pd.merge(result2, df4, on='SpecializationId', how='outer')
            mentor_finalresult.drop(columns=["FieldId", "SpecializationId"], inplace=True)
            # Impute missing values
            mentor_finalresult.dropna(inplace=True)

            # Validate mentor data
            if mentor_finalresult.empty:
                logger.error("No mentors found matching the criteria.")
                raise ValueError("Mentor data is empty.")
            mentor_finalresult.to_csv("final_mentorss_db.csv", index=False)

            # Query Interactions
            query02 = text("SELECT MentorId, MenteeId, Rating FROM Sessions")
            interaction_result = pd.read_sql(query02, connection)
            if interaction_result.empty:
                logger.error("No interactions found.")
                raise ValueError("Interaction data is empty.")
            interaction_result.to_csv("final_interaction_db.csv", index=False)

            # Validate ID consistency (Suggestion 2)
            valid_mentee_ids = set(menteefinalresult['Id'])
            valid_mentor_ids = set(mentor_finalresult['Id'])
            interaction_result = interaction_result[
                interaction_result['MenteeId'].isin(valid_mentee_ids) &
                interaction_result['MentorId'].isin(valid_mentor_ids)
            ]
            interaction_result.dropna(inplace=True)
            if interaction_result.empty:
                logger.error("No valid interactions after ID validation.")
                raise ValueError("Interaction data is empty after validation.")

            return menteefinalresult, mentor_finalresult, interaction_result

    except OperationalError as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise
    except DatabaseError as e:
        logger.error(f"Query error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise
    finally:
        engine.dispose()

# Preprocess data and train model
def preprocess_and_train():
    try:
        logger.info("Starting data preprocessing and model training")
        mentee_df, mentor_df, interaction_df = fetch_data_from_database()
        mentee_data = np.array(list(mentee_df['Id'].values)).reshape(-1, 1)
        mentor_data = np.array(list(mentor_df['Id'].values)).reshape(-1, 1)
        # Define the desired order of categories
        mentee_desired_order = list(mentee_df['Id'].values)
        mentor_desired_order = list(mentor_df['Id'].values)
        # Initialize OrdinalEncoder with the specified category order
        mentee_encoder = OrdinalEncoder(categories=[mentee_desired_order], dtype=int)
        mentor_encoder = OrdinalEncoder(categories=[mentor_desired_order], dtype=int)
        # Fit and transform the data
        mentee_encoder.fit(mentee_data)
        mentor_encoder.fit(mentor_data)
        interaction_df['mentee_idx'] = mentee_encoder.transform(np.array(list(interaction_df['MenteeId'].values)).reshape(-1, 1))
        interaction_df['mentor_idx'] = mentor_encoder.transform(np.array(list(interaction_df['MentorId'].values)).reshape(-1, 1))

        # Create interaction matrix
        interactions = coo_matrix(
            (interaction_df['Rating'].values,
             (interaction_df['mentee_idx'].values, interaction_df['mentor_idx'].values)),
            shape=(len(mentee_df), len(mentor_df))
        ).tocsr()

        # Preprocess mentee features
       # --- Mentee Features ---

        mentee_df['mentee_skills_str'] = mentee_df['About']+ ' at ' + mentee_df['SpecializationName'] +': '+ mentee_df['FieldName']
        # Initialize the sentence transformer model
        try:
            skills_vectorizer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Successfully initialized SentenceTransformer model")
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer: {str(e)}")
            raise

        # Generate embeddings for mentor tracks
        try:
            mentee_skills_matrix = skills_vectorizer.encode(
                mentee_df['mentee_skills_str'].tolist(),
                convert_to_numpy=True,
                show_progress_bar=True
            )
            # Convert to CSR matrix for compatibility with LightFM
            mentee_skills_matrix = csr_matrix(mentee_skills_matrix)
            # print(f"Mentor tracks embedding matrix shape: {mentee_skills_matrix.shape}")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
        mentee_location_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        mentee_location_matrix = mentee_location_encoder.fit_transform(mentee_df[['Location']])
        mentee_features = hstack([mentee_skills_matrix, mentee_location_matrix], format='csr')
        check_finite(mentee_features, "mentee_features")

        # Preprocess mentor features
        # --- Mentor Features ---
        mentor_df['Experience_track_str'] = mentor_df['About'] + ' at ' + mentor_df['SpecializationName'] + ': ' + mentor_df['FieldName']
        # Initialize the sentence transformer model

        try:
            mentor_tracks_matrix = skills_vectorizer.encode(
                mentor_df['Experience_track_str'].tolist(),
                convert_to_numpy=True,
                show_progress_bar=True
            )
            # Convert to CSR matrix for compatibility with LightFM
            mentor_tracks_matrix = csr_matrix(mentor_tracks_matrix)
            logger.info(f"Mentor tracks embedding matrix shape: {mentor_tracks_matrix.shape}")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
        mentor_location_encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        mentor_location_matrix = mentor_location_encoder.fit_transform(mentor_df[['Location']])

        scaler = MinMaxScaler()
        numerical_features = mentor_df[['NumberOfExperience', 'Rate']]
        numerical_features_scaled = scaler.fit_transform(numerical_features)
        numerical_features_matrix = csr_matrix(numerical_features_scaled)
        mentor_features = hstack([mentor_tracks_matrix , mentor_location_matrix, numerical_features_matrix], format='csr')
        check_finite(mentor_features, "mentor_features")

        # Train LightFM model
        model = LightFM(no_components=30, loss='warp', random_state=42)
        model.fit(interactions, user_features=mentee_features, item_features=mentor_features, epochs=30, num_threads=4, verbose=True)

        # Save artifacts
        with open('new_hybird_v2.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('new_mentee_encoder_v2.pkl', 'wb') as f:
            pickle.dump(mentee_encoder, f)
        with open('new_mentor_encoder_v2.pkl', 'wb') as f:
            pickle.dump(mentor_encoder, f)
        with open('new_mentee_features_v2.pkl', 'wb') as f:
            pickle.dump(mentee_features, f)
        with open('new_mentor_features_v2.pkl', 'wb') as f:
            pickle.dump(mentor_features, f)
        with open('new_mentor_df_hybird_v2.pkl', 'wb') as f:
            pickle.dump(mentor_df, f)
        logger.info("Model and artifacts saved successfully")

        return model, mentee_encoder, mentor_encoder, mentee_features, mentor_features, mentor_df

    except Exception as e:
        logger.error(f"Error in preprocessing and training: {str(e)}")
        raise


# Load artifacts
def load_artifacts():
    try:
        # Load the saved model and objects
        with open('new_hybird_v2.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('new_mentee_encoder_v2.pkl', 'rb') as f:
            mentee_encoder = pickle.load(f)
        with open('new_mentor_encoder_v2.pkl', 'rb') as f:
            mentor_encoder = pickle.load(f)
        with open('new_mentee_features_v2.pkl', 'rb') as f:
            mentee_features = pickle.load(f)
        with open('new_mentor_features_v2.pkl', 'rb') as f:
            mentor_features = pickle.load(f)
        with open('new_mentor_df_hybird_v2.pkl', 'rb') as f:
            mentor_df = pickle.load(f)
        with open('new_mentor_tracks_matrix_v2.pkl', 'rb') as f:
            mentor_skills_matrix = pickle.load(f)
        with open('./new_mentor_df_v2_content.pkl', 'rb') as f:
            mentor_df_cont = pickle.load(f)

        logger.info("All model files loaded successfully")
        return model, mentee_encoder, mentor_encoder, mentee_features, mentor_features, mentor_df, mentor_skills_matrix, mentor_df_cont
    except Exception as e:
        logger.error(f"Error loading model files: {str(e)}")
        raise

def get_new_mentee_data(Id):
  engine = create_engine(DATABASE_URL)
  with engine.connect() as connection:
    query1 = text("""
        SELECT Id, Location, About FROM AspNetUsers
        WHERE Id = :id
    """)
    # Execute the query with the parameter and load into a DataFrame
    df = pd.read_sql(query1, connection, params={"id": Id})
    # Query MenteeFields
    query2 = text("SELECT * FROM MenteeFields WHERE MenteeId = :id ")
    df2 = pd.read_sql(query2, connection, params={"id": Id} )
    df2.rename(columns={'MenteeId': 'Id'}, inplace=True)
    result1 = pd.merge(df, df2, on='Id', how='outer')
    f = (result1 ['FieldId'].values)[0]
    # Query Fields
    query3 = text("SELECT * FROM Fields WHERE Id = :id ")
    df3 = pd.read_sql(query3, connection, params={"id": f })
    df3.rename(columns={'Id': "FieldId"}, inplace=True)
    result2 = pd.merge(result1, df3, on='FieldId', how='outer')
    s = (result2 ['SpecializationId'].values)[0]
    # Query Specializations
    query4 = text("SELECT * FROM Specializations where Id  = :id  ")
    df4 = pd.read_sql(query4, connection, params={"id": s } )
    df4.rename(columns={'Id': "SpecializationId", 'Name': "SpecializationName"}, inplace=True)
    # print(df4)
    menteefinalresult = pd.merge(result2, df4, on='SpecializationId', how='outer')
    menteefinalresult.drop(columns=["FieldId", "SpecializationId"], inplace=True)

    # Aggregate and impute missing values
    menteefinalresult = menteefinalresult.groupby('Id').agg({
        'Location': 'first',
        'About': 'first',
        'FieldName': lambda x: ' '.join(x.dropna()),
        'SpecializationName': 'first'
    }).reset_index()

  cont = menteefinalresult['About'] + ' ' + menteefinalresult['SpecializationName'] +' : ' + menteefinalresult['FieldName']
  return (cont.values)[0]

def recommend_mentors_content_based_skills(m_skills, mentor_skills_matrix, mentor_df_cont, top_k=6, weights=None):
 
    mentee_cont = m_skills
    if weights is None:
       weights = {
            'skills': 0.55,
            'experience_years': 0.25,
            'avg_rate': 0.2
        }
    try:
        if not mentee_cont:
          raise ValueError("m_skills cannot be empty")

        encoded_anout = mentor_tracks_vectorizer.encode(
        mentee_cont,
        convert_to_numpy=True,
        show_progress_bar=True
            )

        # Compute similarities
        skill_similarities = cosine_similarity([encoded_anout], mentor_skills_matrix).flatten()
        avg_rate_boost = mentor_df_cont['avg_rate_normalized'].values
        experience_years_boost = mentor_df_cont['experience_years_normalized'].values

        # Combine scores
        combined_scores = (weights['skills'] * skill_similarities)
        final_scores = combined_scores * (1 + weights['avg_rate'] * avg_rate_boost) * (1 + weights['experience_years'] * experience_years_boost)
        top_mentor_indices = np.argsort(-final_scores)[:top_k]
        top_mentor_ids = mentor_df_cont.iloc[top_mentor_indices]['Id'].values
        top_scores = final_scores[top_mentor_indices]

        # Prepare recommendations
        recommendations = []
        for mentor_id, score in zip(top_mentor_ids, top_scores):
            mentor_info = mentor_df_cont[mentor_df_cont['Id'] == mentor_id]
            mentor_details = {
            'mentor_id': str(mentor_id),
            }
            recommendations.append(mentor_details)
        return recommendations

    except Exception as e:
        print("error here")
        return str(e)


# Hypird Recommendation function
def Hypird_model(mentee_id,model, mentee_encoder, mentor_encoder, mentee_features, mentor_features, mentor_df, top_k=5):
    try:
        try:
            mentee_idx = mentee_encoder.transform([[mentee_id]])[0]
        except ValueError:
            logger.error(f"Mentee ID {mentee_id} not found")
            raise ValueError(f"Mentee ID {mentee_id} not found")

        # mentee_idx = mentee_encoder.transform([[test_mentee_id]])[0]
        n_mentees = mentee_features.shape[0]
        if not (0 <= mentee_idx < n_mentees):
            logger.error(f"Mentee_idx {mentee_idx} is out of bounds for matrix with {n_mentees} mentees")
            raise ValueError(f"Mentee_idx {mentee_idx} is out of bounds for matrix with {n_mentees} mentees")

        # Check feature matrices for non-finite values
        check_finite(mentee_features, "mentee_features")
        check_finite(mentor_features, "mentor_features")

        # Generate scores for all mentors
        n_mentors = mentor_features.shape[0]
        mentor_indices = np.arange(n_mentors)
        mentee_indices = np.repeat(mentee_idx, len(mentor_indices))

        scores = model.predict(
                    mentee_indices,
                    mentor_indices,
                    user_features=mentee_features,
                    item_features=mentor_features
                )
        # Get top-k mentor indices and scores
        top_mentor_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_mentor_indices]
        top_mentor_ids = mentor_encoder.inverse_transform(top_mentor_indices.reshape(-1, 1)).flatten()

        # Get mentor details
        recommendations = []
        for mentor_id, score in zip(top_mentor_ids, top_scores):
            mentor_info = mentor_df[mentor_df['Id'] == mentor_id]
            if not mentor_info.empty:
                mentor_details = {
                    'mentor_id': str(mentor_info['Id'].values[0]),
                }
                recommendations.append(mentor_details)
            else:
               logger.warning(f"No info found for mentor ID {mentor_id}")
        return recommendations

    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise


# Scheduler for periodic model updates
def schedule_model_updates(interval_minutes=1440):
    scheduler = BackgroundScheduler()
    scheduler.add_job(preprocess_and_train, 'interval', minutes=interval_minutes)
    try:
        scheduler.start()
        logger.info(f"Scheduled model updates every {interval_minutes} minutes")
    except Exception as e:
        logger.error(f"Failed to start scheduler: {str(e)}")
        raise
    return scheduler

# Flask endpoint for recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        if not data or 'mentee_id' not in data:
            logger.error("Invalid request: mentee_id is required")
            return jsonify({'error': 'mentee_id is required'}), 400
        mentee_id = data['mentee_id']

        menteefinalresult, mentor_finalresult, interaction_df = fetch_data_from_database()
        model, mentee_encoder, mentor_encoder, mentee_features, mentor_features, mentor_df, mentor_skills_matrix, mentor_df_cont = load_artifacts()

        a = list(menteefinalresult['Id'].values)

        if mentee_id in  a:
             recomendationss =  Hypird_model(mentee_id, model, mentee_encoder, mentor_encoder, mentee_features, mentor_features, mentor_df)
        else:
             mentee_cont = get_new_mentee_data(mentee_id)
             recomendationss = recommend_mentors_content_based_skills(mentee_cont, mentor_skills_matrix, mentor_df_cont)

        return jsonify(recomendationss)

    except ValueError as ve:
        logger.error(f"ValueError: {str(ve)}")
        return {'error': str(ve)}, 404
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {'error': 'Internal server error'}, 500

if __name__ == "__main__":
    scheduler = None
    try:
        # Start scheduler
        scheduler = schedule_model_updates(interval_minutes=1440)
        # Run Flask app
        app.run(port=5000)

    except Exception as e:
        logger.error(f"Error starting Flask app: {str(e)}")
        if scheduler and scheduler.running:
            scheduler.shutdown()
            logger.info("Scheduler shut down")
        # sys.exit(1)
    finally:
        if scheduler and scheduler.running:
            scheduler.shutdown()
            logger.info("Scheduler shut down")
