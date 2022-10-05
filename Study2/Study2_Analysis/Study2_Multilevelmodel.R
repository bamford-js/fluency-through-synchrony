"

"

code_dir = dirname(rstudioapi::getActiveDocumentContext()$path)
setwd(code_dir)

data = read.csv('data_filtered_2022-04-20.csv', sep=',')

library(lme4)
library(lmerTest)
library(ggplot2)

data$Participant = factor(data$Participant)
data$Condition = factor(data$Condition)

m1 <- lmer(Performance ~ 1 + Condition * Beat.Variance + (1 | Participant), data = data)

m2 <- lmer(Pupil.Size ~ 1 + Condition + (1 | Participant) + (1 | Trial), data = data)

m3 <- lmer(Beat.Variance ~ 1 + Condition + (1 | Participant), data = data)


summary(m1)
summary(m2)
summary(m3)

library(sjPlot)

plot_model(m1, type = 'pred', terms = c('Beat.Variance', 'Condition'))

plot_model(m3, type = 'pred', terms = c('Condition'))
