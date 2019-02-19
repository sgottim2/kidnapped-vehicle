/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::uniform_int_distribution;
using std::uniform_real_distribution;
using std::cout;
using std::endl;
using std::numeric_limits;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  
  num_particles = 100;  
  default_random_engine gen;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  
  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    particles.push_back(particle);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  default_random_engine gen;
  for (int i = 0; i < num_particles; i++) {     
    double new_x;
    double new_y;
    double new_theta;
    if (fabs(yaw_rate) > 0) {
      new_x = particles[i].x + (velocity/yaw_rate)*(sin(particles[i].theta + (yaw_rate*delta_t)) - sin(particles[i].theta));
      new_y = particles[i].y + (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate*delta_t)));
      new_theta = particles[i].theta + yaw_rate*delta_t;
    } else {
      new_x = particles[i].x + velocity*delta_t*cos(particles[i].theta);
      new_y = particles[i].y + velocity*delta_t*sin(particles[i].theta);
      new_theta = particles[i].theta;
    }
    normal_distribution<double> dist_x(new_x, std_pos[0]);
    normal_distribution<double> dist_y(new_y, std_pos[1]);
    normal_distribution<double> dist_theta(new_theta, std_pos[2]);
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);  
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
 
  for (unsigned int i = 0; i < observations.size(); i++) {
    double min_dist = numeric_limits<double>::max();
    int associated_id;
    LandmarkObs obs = observations[i];
    for (unsigned int j = 0; j < predicted.size(); j++) {
      LandmarkObs predict = predicted[j];
      double current_dist = dist(obs.x, obs.y, predict.x, predict.y);
      if (current_dist < min_dist) {
        min_dist = current_dist;
        associated_id = predict.id;
      }
    }
    observations[i].id = associated_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  weights.clear();
  for (int i = 0; i < num_particles; i++) {        
    double particle_x = particles[i].x;
    double particle_y = particles[i].y;
    double particle_theta = particles[i].theta;
        
    vector<LandmarkObs> predicted;
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      int landmark_id = map_landmarks.landmark_list[j].id_i;
      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;
      double d = dist(particle_x,particle_y,landmark_x,landmark_y);
      if (d<=sensor_range) {
        predicted.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
      }
    }
    
    vector<LandmarkObs> transformed_obs;    
    for (unsigned int j = 0; j < observations.size(); j++) {
      int transformed_id = observations[j].id;
      double transformed_x = cos(particle_theta) * observations[j].x - sin(particle_theta) * observations[j].y + particle_x;
      double transformed_y = sin(particle_theta) * observations[j].x + cos(particle_theta) * observations[j].y + particle_y;
      transformed_obs.push_back(LandmarkObs{transformed_id, transformed_x, transformed_y});
    }

    dataAssociation(predicted, transformed_obs);
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;
    for (unsigned int j = 0; j < transformed_obs.size(); j++) {
      associations.push_back(transformed_obs[j].id);
      sense_x.push_back(transformed_obs[j].x);
      sense_y.push_back(transformed_obs[j].y);
    }
    SetAssociations(particles[i], associations, sense_x, sense_y);
    particles[i].weight = 1.0;
    
    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];
    double gauss_norm = 1.0 / (2.0 * M_PI * sig_x * sig_y);

    
    for (unsigned int j = 0; j < transformed_obs.size(); j++) {
      double obs_x = transformed_obs[j].x;
      double obs_y = transformed_obs[j].y;
      double predicted_x, predicted_y;
      for (unsigned int k = 0; k < predicted.size(); k++) {
        if (predicted[k].id == transformed_obs[j].id) {
          predicted_x = predicted[k].x;
          predicted_y = predicted[k].y;
        }
      }         
      double obs_weight = gauss_norm*exp(-1.0*( (pow(obs_x - predicted_x,2)/(2.0*sig_x*sig_x)) +  (pow(obs_y - predicted_y,2)/(2.0*sig_x*sig_x))   ));

      particles[i].weight *= obs_weight;
    }
    weights.push_back(particles[i].weight);
  }
}


void ParticleFilter::resample() {
  vector<Particle> new_particles;
  default_random_engine gen;
  uniform_int_distribution<int> d(0,num_particles-1);
  int index = d(gen);
  double beta = 0.0;
  double mw = *max_element(weights.begin(), weights.end());
  uniform_real_distribution<double> dD(0.0, mw);
  for(int j=0;j<num_particles;j++){
    beta += dD(gen) * 2.0;
    while(beta>weights[index]){
      beta -= weights[index];
      index = (index+1)%num_particles;
    }
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
}


void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}