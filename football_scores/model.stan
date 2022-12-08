data {
  int<lower=0> N;       // number of games
  int<lower=0> M;       // number of players per team
  int<lower=0> total_players; // total number of players
  array[N] int y;           // difference in goals scored for each game (team1_goals - team2_goals)
  array[N, M] int team1;    // indices of players in team 1
  array[N, M] int team2;    // indices of players in team 2

  array[2] real p_skills;   // Get priors as data so that 
  real p_beta;              // model does not need to be rebuilt
  real p_nu;
  real p_theta;
}

parameters {
  array[total_players] real<lower=0, upper=10> skills; // skill levels of team 1 players
  real<lower=0> beta;  // 1/scale
  real<lower=0> nu;    // degrees of freedom
  real<lower=0> theta; // 1 skill advatage moves expected goall diff by 1
}

model {
  skills ~ normal(p_skills[1], p_skills[2]); // prior distribution for skills
  beta ~ exponential(p_beta); // prior distribution for precision parameter
  nu ~ exponential(p_nu);
  theta ~ exponential(p_theta);

  for (i in 1:N) { 
    real diff = sum(skills[team1[i]]) - sum(skills[team2[i]]); // difference in team skills
    y[i] ~ student_t(nu, diff * theta, 1/beta); // likelihood function
  }
}