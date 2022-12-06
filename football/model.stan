data {
    int N;                  // Games played
    int M;                  // Players in a team
    array[N, M] int winners;
    array[N, M] int losers;
}

parameters {
    array[2*M] real<lower=0, upper=10> x;
}

model {
    real p[N];
    real skill_w;
    real skill_l;

    x ~ uniform(0, 10);

    for (i in 1:N) {
        skill_w = 0;         // Total skill of winning team
        skill_l = 0;         // Total skill of losing team

        for (j in 1:M) {
            skill_w = skill_w + x[winners[i][j]];
            skill_l = skill_l + x[losers[i][j]];
        }

        p[i] = inv_logit(0.3 * (skill_w - skill_l));
        1 ~ bernoulli(p[i]);
    }
}
