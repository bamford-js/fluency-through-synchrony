"

"

code_dir = dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(code_dir)

data = read.csv('data_filtered_2022-05-30.csv', sep=',')

library(lme4)
library(lmerTest)
library(MuMIn)
library(sjPlot)

data$Participant = factor(data$Participant)
data$Drums = factor(data$Drums)
data$Tempo.Ratio = ordered(data$Tempo.Ratio, levels = c('120-080','120-113','120-120')) # c('120-120','120-080','120-113')) for graph #  for model
contrasts(data$Tempo.Ratio) = contr.sum(3)


# maybe re-scale variables?
# Bonding and Difficulty are 1-100. Pupil Size is ~3-6. IPA is ~3-6. Beat Variance is ~0.001-0.05


m1 = lmer(Bonding ~ 1 + Tempo.Ratio + Tempo.Ratio*Difficulty + Tempo.Ratio*Pupil.Size + Difficulty*Pupil.Size + (1 | Participant), data = data)
summary(m1)
# this is the good one
m = lmer(Bonding ~ 1 + Tempo.Ratio*Drums + (1 | Participant), data = data)
summary(m)
r.squaredGLMM(m)
tab_model(m1)



m2 = lmer(Difficulty ~ 1 + Tempo.Ratio + (1 | Participant), data = data)
m3 = lmer(Pupil.Size ~ 1 + Tempo.Ratio + (1 | Participant), data = data)
m4 = lmer(Bonding ~ 1 + Difficulty + (1 | Participant), data = data)
m5 = lmer(Bonding ~ 1 + Pupil.Size + (1 | Participant), data = data)

m6 = lmer(Bonding ~ 1 + Tempo.Ratio + (1 | Participant), data = data)

m7 = lmer(Bonding ~ 1 + Difficulty*Tempo.Ratio + (1 | Participant), data = data)
summary(m7)

m8 = lmer(Bonding ~ 1 + Tempo.Ratio + Difficulty + (1 | Participant), data = data)
summary(m8)

# what to do about singular fit? Can't reduce the number of variables much more...

# check lmer uses for comparisson level in ordered variable


summary(m2)
summary(m3)
summary(m4)
summary(m5)
summary(m6)




plot_model(m1, type = 'pred', terms = c('Pupil.Size', 'Difficulty [quart]]'), title = NULL) #[['data']]
plot_model(m, type = 'pred', terms = c('Tempo.Ratio', 'Drums'), title = NULL) #[['data']]

library(mediation)
set.seed(2022)
detach("package:lmerTest", unload=TRUE)

#mediation relationship: Tempo -> Difficulty -> Bonding
# could add covariates, but haven't
# are 3 levels of tempo a problem?

data80 <- data[data$X120.113 != 1,]
data113 <- data[data$X120.080 != 1,]




# Mediation for 120-80
# Effect of TR on Bonding
direct = lmer(Bonding ~ X120.120 + (1 | Participant), data = data80)
summary(direct)

med_fit = lmer(Difficulty ~ X120.120 + (1 | Participant), data = data80)
# Effect of TR on Difficulty
summary(med_fit)
r.squaredGLMM(med_fit)

out_fit = lmer(Bonding ~ Difficulty + X120.120 + (1 | Participant), data = data80)
# Effect of TR and Difficulty on Bonding
summary(out_fit)
r.squaredGLMM(out_fit)

med_out = mediate(med_fit, out_fit, treat = "X120.120", mediator = "Difficulty", sims = 1000, boot.ci.type = "bca")
# Effect of TR on Bonding, mediated by Difficulty
summary(med_out)

# Mediation for 120-113
# Effect of TR on Bonding
direct = lmer(Bonding ~ X120.120 + (1 | Participant), data = data113)
summary(direct)

med_fit = lmer(Difficulty ~ X120.120 + (1 | Participant), data = data113)
# Effect of TR on Difficulty
summary(med_fit)
r.squaredGLMM(med_fit)

out_fit = lmer(Bonding ~ Difficulty + X120.120 + (1 | Participant), data = data113)
# Effect of TR and Difficulty on Bonding
summary(out_fit)
r.squaredGLMM(out_fit)

med_out = mediate(med_fit, out_fit, treat = "X120.120", mediator = "Difficulty", sims = 1000, boot.ci.type = "bca")
# Effect of TR on Bonding, mediated by Difficulty
summary(med_out)




# pupil size

# Mediation for 120-80
# Effect of TR on Bonding
direct = lmer(Bonding ~ X120.120 + (1 | Participant), data = data80)
summary(direct)

med_fit = lmer(Pupil.Size ~ X120.120 + (1 | Participant), data = data80)
# Effect of TR on Difficulty
summary(med_fit)
r.squaredGLMM(med_fit)

out_fit = lmer(Bonding ~ Pupil.Size + X120.120 + (1 | Participant), data = data80)
# Effect of TR and Difficulty on Bonding
summary(out_fit)
r.squaredGLMM(out_fit)

med_out = mediate(med_fit, out_fit, treat = "X120.120", mediator = "Pupil.Size", sims = 1000, boot.ci.type = "bca")
# Effect of TR on Bonding, mediated by Difficulty
summary(med_out)

# Mediation for 120-113
# Effect of TR on Bonding
direct = lmer(Bonding ~ X120.120 + (1 | Participant), data = data113)
summary(direct)

med_fit = lmer(Pupil.Size ~ X120.120 + (1 | Participant), data = data113)
# Effect of TR on Difficulty
summary(med_fit)
r.squaredGLMM(med_fit)

out_fit = lmer(Bonding ~ Pupil.Size + X120.120 + (1 | Participant), data = data113)
# Effect of TR and Difficulty on Bonding
summary(out_fit)
r.squaredGLMM(out_fit)

med_out = mediate(med_fit, out_fit, treat = "X120.120", mediator = "Pupil.Size", sims = 1000, boot.ci.type = "bca")
# Effect of TR on Bonding, mediated by Difficulty
summary(med_out)

