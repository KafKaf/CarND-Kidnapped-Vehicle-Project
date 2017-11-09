/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <sstream>
#include <string>
#include <iterator>
#include <unordered_map>

#include "particle_filter.h"
#include "map.h"

using namespace std;

// Random engine to sample from our normal distributed variables
static default_random_engine gen;
// Normal distribution - zero mean - measurements noise
//TODO: Change if std of init(GPS) is different from std of prediction(all measurements)
static normal_distribution<double> dist_zero_mean_x_noise;
static normal_distribution<double> dist_zero_mean_y_noise;
static normal_distribution<double> dist_zero_mean_theta_noise;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // set number of particles
    num_particles = 100;

    // default start weight
    double weight = 1.0;

    // initialize normal distribution - zero mean - measurements noise
    dist_zero_mean_x_noise = normal_distribution<double>(0, std[0]);
    dist_zero_mean_y_noise = normal_distribution<double>(0, std[1]);
    dist_zero_mean_theta_noise = normal_distribution<double>(0, std[2]);

    for (int i = 0;i < num_particles;i++) {
        // sample randomly from normal distributions initialized earlier with the initial state
        double init_x = x + dist_zero_mean_x_noise(gen);
        double init_y = y + dist_zero_mean_y_noise(gen);
        double init_theta = theta + dist_zero_mean_theta_noise(gen);

        Particle particle = {
            i, // id
            init_x, // x
            init_y, // y
            init_theta, // theta
            weight // weight
        };

        // add particle to particles vector
        particles.push_back(particle);

        // add particle weight to particles weights vector
        weights.push_back(weight);
    }

    // set initialization flag to true
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // predict new particles state from measurements with random Guassian noise
    // yaw rate > 0
    if (fabs(yaw_rate) > 0.001) {
        double yaw_rate_multiply_delta_t = yaw_rate * delta_t;
        double velocity_divide_yaw_rate = velocity / yaw_rate;
        for (int i = 0;i < particles.size();i++) {
            particles[i].x += dist_zero_mean_x_noise(gen) + velocity_divide_yaw_rate
                * (sin(particles[i].theta + yaw_rate_multiply_delta_t) - sin(particles[i].theta));
            particles[i].y += dist_zero_mean_y_noise(gen) + velocity_divide_yaw_rate
                * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate_multiply_delta_t));
            particles[i].theta += dist_zero_mean_theta_noise(gen) + yaw_rate_multiply_delta_t;
        }
    // yaw rate = 0
	} else {
        double velocity_multiply_delta_t = velocity * delta_t;
        for (int i = 0;i < particles.size();i++) {
            particles[i].x += dist_zero_mean_x_noise(gen) + velocity_multiply_delta_t * cos(particles[i].theta);
            particles[i].y += dist_zero_mean_y_noise(gen) + velocity_multiply_delta_t * sin(particles[i].theta);
        }
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	for (int i = 0;i < observations.size();i++) {
	    double min_distance = dist(observations[i].x, observations[i].y, predicted[0].x, predicted[0].y);
	    int min_distance_id = predicted[0].id;
	    for (int j=1;j < predicted.size();j++) {
            double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
            if (distance < min_distance) {
                min_distance = distance;
                min_distance_id = predicted[j].id;
            }
	    }
	    observations[i].id = min_distance_id;
	}
}

void ParticleFilter::transformObservationsFromVehicleToMapCoordinates(const std::vector<LandmarkObs>& observations
    , std::vector<LandmarkObs> &landmark_observations_map, Particle &particle) {
    for (int i = 0;i < observations.size();i++) {
        LandmarkObs landmark_observation_map;
        // translation and rotation by position and bearing
        landmark_observation_map.x = particle.x + cos(particles[i].theta) * observations[i].x - sin(particle.theta) * observations[i].y;
        landmark_observation_map.y = particle.y + cos(particles[i].theta) * observations[i].y + sin(particle.theta) * observations[i].x;
        landmark_observations_map.push_back(landmark_observation_map);
    }
}

void ParticleFilter::convertAndFilterLandmarksByRange(double sensor_range, const Map &map_landmarks
    , std::vector<LandmarkObs> &landmark_predictions_map, Particle &particle) {

    for (int i = 0;i < map_landmarks.landmark_list.size();i++) {
        // Euclidean distance function is computationally expensive, so just run faster and less accurate check
        if (fabs(map_landmarks.landmark_list[i].x_f - particle.x) <= sensor_range
            && fabs(map_landmarks.landmark_list[i].y_f - particle.y) <= sensor_range) {

            LandmarkObs landmark;
            landmark.id = map_landmarks.landmark_list[i].id_i;
            landmark.x = map_landmarks.landmark_list[i].x_f;
            landmark.y = map_landmarks.landmark_list[i].y_f;

            landmark_predictions_map.push_back(landmark);
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

    // update weights for each particle
	for (int i = 0;i < particles.size();i++) {
	    // vectors to be modified by other functions
	    vector<LandmarkObs> landmark_observations_map;
	    vector<LandmarkObs> landmark_predictions_map;

        // transform observations from vehicle to map point of view
        transformObservationsFromVehicleToMapCoordinates(observations, landmark_observations_map, particles[i]);
        // filter out of range landmarks and convert to landmark observations type
	    convertAndFilterLandmarksByRange(sensor_range, map_landmarks, landmark_predictions_map, particles[i]);
        // associate observation with nearest landmark
	    dataAssociation(landmark_predictions_map, landmark_observations_map);

        // set final weight to 1 so won't affect multiplications
        double final_weight = 1;

	    for (int j = 0;j < landmark_observations_map.size();j++) {
            // get associated landmark map position
            int landmark_id = landmark_observations_map[j].id;
            auto t = map_landmarks.landmark_hashtable.find(landmark_id);
            // not being defensive here
            Map::single_landmark_s landmark = t->second;
            double prediction_x = landmark.x_f;
            double prediction_y = landmark.y_f;

            // calculate measurements multi-variate probability(guass_norm*exp(-exponent))
	        // calculate normalization term
	        double gauss_norm = (1/(2 * M_PI * std_landmark[0] * std_landmark[1]));
	        // calculate exponent
            double exponent = pow(landmark_observations_map[j].x - prediction_x,2)/pow(2 * std_landmark[0],2)
                + pow(landmark_observations_map[j].y - prediction_y,2)/pow(2 * std_landmark[1],2);

            // particles final weight will be calculated as the product of each measurement's Multivariate-Gaussian probability
            final_weight *= gauss_norm * exp(-exponent);
	    }

	    // set particle weight
	    particles[i].weight = final_weight;
	    weights[i] = final_weight;
	}
}

void ParticleFilter::resample() {
    // initialize a random distribution where the probability of each element is defined by weights of particles.
    discrete_distribution<> d(weights.begin(), weights.end());
    // new temp vector of particles after re-sampling
    vector<Particle> resampled_particles;

    // sample particles randomly from discrete with probability proportional to their weight.
    for (int i = 0;i < particles.size();i++) {
        resampled_particles.push_back(particles[d(gen)]);
    }
    // swap particles vectors
    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
