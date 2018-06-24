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
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // Set the number of particles. Initialize all particles to first position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  default_random_engine gen;
  double std_x, std_y, std_theta;

  num_particles = 100;

  /**
   * Create normal distributions
   */
  std_x = std[0];
  std_y = std[1];
  std_theta = std[2];
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;

    particles.push_back(p);
    weights.push_back(1);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/
  default_random_engine gen;
  double std_x, std_y, std_theta;

  std_x = std_pos[0];
  std_y = std_pos[1];
  std_theta = std_pos[2];

  for (auto &p : particles) {
    double new_x;
    double new_y;
    double new_theta;

    if (yaw_rate == 0) {
      new_x = p.x + velocity * delta_t * cos(p.theta);
      new_y = p.y + velocity * delta_t * sin(p.theta);
      new_theta = p.theta;
    } else {
      /**
       * 7 and 8 of Lesson 14
       */
      new_x = p.x + velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
      new_y = p.y + velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
      new_theta = p.theta + yaw_rate * delta_t;
    }

    normal_distribution<double> dist_x(new_x, std_x);
    normal_distribution<double> dist_y(new_y, std_y);
    normal_distribution<double> dist_theta(new_theta, std_theta);

    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
  // Update the weights of each particle using a mult-variate Gaussian distribution. You can read
  //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
  //   according to the MAP'S coordinate system. You will need to transform between the two systems.
  //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
  //   The following is a good resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement (look at equation
  //   3.33
  //   http://planning.cs.uiuc.edu/node99.html
  double sig_x = std_landmark[0];
  double sig_y = std_landmark[1];
  double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  int idx = 0;
  for (auto &p : particles) {
    p.weight = 1.0;

    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    /**
     * Homogenous Transformation - Part 15 of Lesson 14
     * xm ​ =xp ​+ (cosθ × xc) − (sinθ × yc)
     * ym ​ =yp ​+ (sinθ × xc) + (cosθ × yc)
     */
    for (auto obs : observations) {
      LandmarkObs trans_obs = {};
      trans_obs.x = p.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
      trans_obs.y = p.y + (sin(p.theta) * obs.x) + (cos(p.theta) * obs.y);

    /**
     * Find the nearest landmark associate to
     * each observation of current particle.
     */
      double nearest_distance = sensor_range;
      int association = 0;

      for (auto map_landmark : map_landmarks.landmark_list) {
        double lm_x = map_landmark.x_f;
        double lm_y = map_landmark.y_f;

        double distance = sqrt(
            pow(trans_obs.x - lm_x, 2) + pow(trans_obs.y - lm_y, 2)
        );

        if (distance < nearest_distance) {
          nearest_distance = distance;
          association = map_landmark.id_i;
        }
      }

      /**
       * Using Multivariate-Gaussian Probability to update
       * particle's weight.
       */
      if (association != 0) {
        double mu_x = map_landmarks.landmark_list[association-1].x_f;
        double mu_y = map_landmarks.landmark_list[association-1].y_f;

        double exponent = -(pow(trans_obs.x - mu_x, 2) / (2 * sig_x * sig_x) +
            pow(trans_obs.y - mu_y, 2) / (2 * sig_y * sig_y));
        long double w = gauss_norm * exp(exponent);
        if (w > 0) {
          p.weight *= w;
        }
      } else {
        p.weight = 0;
      }

      associations.push_back(association);
      sense_x.push_back(trans_obs.x);
      sense_y.push_back(trans_obs.y);
    }

    SetAssociations(p, associations, sense_x, sense_y);
    weights[idx] = p.weight;
    idx++;
  }
}

void ParticleFilter::resample() {
  // Resample particles with replacement with probability proportional to their weight.
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  default_random_engine gen;
  discrete_distribution<int> distribution(weights.begin(), weights.end());

  vector<Particle> resample_particles;
  for (int i = 0; i < num_particles; i++) {
    resample_particles.push_back(particles[distribution(gen)]);
  }

  particles = resample_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
  //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

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
