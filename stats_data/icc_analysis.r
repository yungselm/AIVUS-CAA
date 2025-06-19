library(tidyverse)
library(ggplot2)
library(psych)
library(irr)

setwd("D:/NIVA_analysis/NIVA_analysis")
# light mode
create_bland_altman_plot <- function(df, col_A, col_B, y_bound_upper=NA, y_bound_lower=NA) {
  
  # Create new column for average measurement
  df$avg <- rowMeans(df[, c(col_A, col_B)])
  
  # Create new column for difference in measurements
  df$diff <- df[[col_A]] - df[[col_B]]
  
  # Calculate mean difference and standard deviation
  mean_diff <- mean(df$diff)
  lower <- mean_diff - 1.96 * sd(df$diff)
  upper <- mean_diff + 1.96 * sd(df$diff)
  
  # Create the Bland-Altman plot
  plot <- ggplot(df, aes(x = avg, y = diff)) +
    geom_point(size = 2) +
    geom_hline(yintercept = mean_diff, color = "blue", linetype = "solid") +
    geom_hline(yintercept = lower, color = "red", linetype = "dashed") +
    geom_hline(yintercept = upper, color = "red", linetype = "dashed") +
    ggtitle("Bland-Altman Plot") +
    ylab("Difference Between Measurements") +
    xlab("Average Measurement") +
    theme_minimal()
  
  if (!is.na(y_bound_upper) && !is.na(y_bound_lower)) {
    plot <- plot + ylim(y_bound_lower, y_bound_upper)
  }
  
  # Return the plot
  return(plot)
}

# patient data gating by eye or with tool intra-observer
data <- read.csv("icc.csv", sep=";")
niva <- data %>% select("frame_niva_run1", "frame_niva_run2")
ba_niva <- create_bland_altman_plot(niva, "frame_niva_run1", "frame_niva_run2", 3, -3)

ICC(niva)

eye <- data %>% select("frame_eye_run1", "frame_eye_run2")
ba_eye <- create_bland_altman_plot(eye, "frame_eye_run1", "frame_eye_run2", 3, -3) 

ICC(eye)


## Gating
gating_rest <- read.csv("output/icc_gating_rest.csv")
gating_rest <- gating_rest %>% select("ivus_frame", "frame")
ICC(gating_rest)
ba_gating_rest <- create_bland_altman_plot(gating_rest, "ivus_frame", "frame", 7.5, -7.5)

gating_stress <- read.csv("output/icc_gating_stress.csv")
gating_stress <- gating_stress %>% select("ivus_frame", "frame")
ICC(gating_stress)
ba_gating_stress <- create_bland_altman_plot(gating_stress, "ivus_frame", "frame", 7.5, -7.5)

gating_rest_eye <- read.csv("output/icc_gating_rest_eye.csv")
gating_rest_eye <- gating_rest_eye %>% select("ivus_frame", "frame")
ICC(gating_rest_eye)
ba_gating_rest_eye <- create_bland_altman_plot(gating_rest_eye, "ivus_frame", "frame", 7.5, -7.5)

gating_stress_eye <- read.csv("output/icc_gating_stress_eye.csv")
gating_stress_eye <- gating_stress_eye %>% select("ivus_frame", "frame")
ICC(gating_stress_eye)
ba_gating_stress_eye <- create_bland_altman_plot(gating_stress_eye, "ivus_frame", "frame", 7.5, -7.5)

# save all plots
ggsave("output/ba_niva.png", ba_niva, width = 6, height = 4)
ggsave("output/ba_eye.png", ba_eye, width = 6, height = 4)
ggsave("output/ba_gating_rest.png", ba_gating_rest, width = 6, height = 4)
ggsave("output/ba_gating_stress.png", ba_gating_stress, width = 6, height = 4)
ggsave("output/ba_gating_rest_eye.png", ba_gating_rest_eye, width = 6, height = 4)
ggsave("output/ba_gating_stress_eye.png", ba_gating_stress_eye, width = 6, height = 4)


##################################################
# ICC interreader analysis
# read .txt file input/NARCO200_rest_AS.txt
narco200_rest_R1 <- read.table("input/NARCO200_rest_AS.txt", header=TRUE, sep="\t")
narco200_rest_R2 <- read.table("input/NARCO200_rest_MB.txt", header=TRUE, sep="\t")
narco200_stress_R1 <- read.table("input/NARCO200_stress_AS.txt", header=TRUE, sep="\t")
narco200_stress_R2 <- read.table("input/NARCO200_stress_MB.txt", header=TRUE, sep="\t")

narco200_rest_R1 <- narco200_rest_R1 %>% filter(phase != "-")
narco200_rest_R2 <- narco200_rest_R2 %>% filter(phase != "-")
print(narco200_rest_R1$frame[1])
print(narco200_rest_R2$frame[1])
narco200_rest_R2 <- narco200_rest_R2 %>% filter(frame >= narco200_rest_R1$frame[1])
nrow(narco200_rest_R1)
nrow(narco200_rest_R2)
# remove last two rows from narcos200_rest_R2
narco200_rest_R2 <- narco200_rest_R2[-c(nrow(narco200_rest_R2), nrow(narco200_rest_R2)-1),]

narco200_stress_R1 <- narco200_stress_R1 %>% filter(phase != "-")
narco200_stress_R2 <- narco200_stress_R2 %>% filter(phase != "-")
print(narco200_stress_R1$frame[1])
print(narco200_stress_R2$frame[1])
narco200_stress_R1 <- narco200_stress_R1 %>% filter(frame >= narco200_stress_R2$frame[1])
nrow(narco200_stress_R1)
nrow(narco200_stress_R2)

gating_rest_inter <- data.frame(narco200_rest_R1$frame, narco200_rest_R2$frame)
gating_stress_inter <- data.frame(narco200_stress_R1$frame, narco200_stress_R2$frame)

ICC(gating_rest_inter)
ba_gating_rest_inter <- create_bland_altman_plot(gating_rest_inter, "narco200_rest_R1.frame", "narco200_rest_R2.frame", 7.5, -7.5)

ICC(gating_stress_inter)
ba_gating_stress_inter <- create_bland_altman_plot(gating_stress_inter, "narco200_stress_R1.frame", "narco200_stress_R2.frame", 7.5, -7.5)

ggsave("output/ba_gating_rest_inter.png", ba_gating_rest_inter, width = 6, height = 4)
ggsave("output/ba_gating_stress_inter.png", ba_gating_stress_inter, width = 6, height = 4)

# segmentation
NARCO_216_rest_R1 <- read.table("input/NARCO_216_rest_R1.txt", header=TRUE, sep="\t")
NARCO_216_rest_R2 <- read.table("input/NARCO_216_rest_R2.txt", header=TRUE, sep="\t")
NARCO_216_rest_R1 <- NARCO_216_rest_R1 %>% filter((frame >= 15 & frame <= 30) | (frame >=1210 & frame <= 1225))
NARCO_216_rest_R2 <- NARCO_216_rest_R2 %>% filter((frame >= 15 & frame <= 30) | (frame >=1210 & frame <= 1225))

NARCO_218_rest_R1 <- read.table("input/NARCO_218_rest_R1.txt", header=TRUE, sep="\t")
NARCO_218_rest_R2 <- read.table("input/NARCO_218_rest_R2.txt", header=TRUE, sep="\t")
NARCO_218_rest_R1 <- NARCO_218_rest_R1 %>% filter((frame >= 1 & frame <= 15) | (frame >= 430 & frame <= 445))
NARCO_218_rest_R2 <- NARCO_218_rest_R2 %>% filter((frame >= 1 & frame <= 15) | (frame >= 430 & frame <= 445))

NARCO_234_rest_R1 <- read.table("input/NARCO_234_rest_R1.txt", header=TRUE, sep="\t")
NARCO_234_rest_R2 <- read.table("input/NARCO_234_rest_R2.txt", header=TRUE, sep="\t")
NARCO_234_rest_R1 <- NARCO_234_rest_R1 %>% filter((frame >= 1 & frame <= 15) | (frame >= 1000 & frame <= 1015))
NARCO_234_rest_R2 <- NARCO_234_rest_R2 %>% filter((frame >= 1 & frame <= 15) | (frame >= 1000 & frame <= 1015))

NARCO_216_stress_R1 <- read.table("input/NARCO_216_stress_R1.txt", header=TRUE, sep="\t")
NARCO_216_stress_R2 <- read.table("input/NARCO_216_stress_R2.txt", header=TRUE, sep="\t")
NARCO_216_stress_R1 <- NARCO_216_stress_R1 %>% filter((frame >= 300 & frame <= 315) | (frame >= 1650 & frame <= 1665))
NARCO_216_stress_R2 <- NARCO_216_stress_R2 %>% filter((frame >= 300 & frame <= 315) | (frame >= 1650 & frame <= 1665))

NARCO_218_stress_R1 <- read.table("input/NARCO_218_stress_R1.txt", header=TRUE, sep="\t")
NARCO_218_stress_R2 <- read.table("input/NARCO_218_stress_R2.txt", header=TRUE, sep="\t")
NARCO_218_stress_R1 <- NARCO_218_stress_R1 %>% filter((frame >= 1 & frame <= 15) | (frame >= 960 & frame <= 975))
NARCO_218_stress_R2 <- NARCO_218_stress_R2 %>% filter((frame >= 1 & frame <= 15) | (frame >= 960 & frame <= 975))

NARCO_234_stress_R1 <- read.table("input/NARCO_234_stress_R1.txt", header=TRUE, sep="\t")
NARCO_234_stress_R2 <- read.table("input/NARCO_234_stress_R2.txt", header=TRUE, sep="\t")
NARCO_234_stress_R1 <- NARCO_234_stress_R1 %>% filter((frame >= 1 & frame <= 15) | (frame >= 1200 & frame <= 1215))
NARCO_234_stress_R2 <- NARCO_234_stress_R2 %>% filter((frame >= 1 & frame <= 15) | (frame >= 1200 & frame <= 1215))

NARCO_218_rest_R1 <- NARCO_218_rest_R1 %>% select(-c(pullback_speed, pullback_start_frame, frame_rate))
reader1_rest <- rbind(NARCO_216_rest_R1, NARCO_218_rest_R1, NARCO_234_rest_R1)
reader2_rest <- rbind(NARCO_216_rest_R2, NARCO_218_rest_R2, NARCO_234_rest_R2)

area_rest <- data.frame(reader1_rest$lumen_area, reader2_rest$lumen_area)
shortest_distance_rest <- data.frame(reader1_rest$shortest_distance, reader2_rest$shortest_distance)

ICC(area_rest)
ICC(shortest_distance_rest)

ba_area_rest <- create_bland_altman_plot(area_rest, "reader1_rest.lumen_area", "reader2_rest.lumen_area", 4.5, -4.5)
ba_shortest_distance_rest <- create_bland_altman_plot(shortest_distance_rest, "reader1_rest.shortest_distance", "reader2_rest.shortest_distance", 1, -1)

ggsave("output/ba_area_rest.png", ba_area_rest, width = 6, height = 4)
ggsave("output/ba_shortest_distance_rest.png", ba_shortest_distance_rest, width = 6, height = 4)

NARCO_216_stress_R1 <- NARCO_216_stress_R1 %>% select(-c(pullback_speed, pullback_start_frame, frame_rate))
reader1_stress <- rbind(NARCO_216_stress_R1, NARCO_218_stress_R1, NARCO_234_stress_R1)
reader2_stress <- rbind(NARCO_216_stress_R2, NARCO_218_stress_R2, NARCO_234_stress_R2)

area_stress <- data.frame(reader1_stress$lumen_area, reader2_stress$lumen_area)
shortest_distance_stress <- data.frame(reader1_stress$shortest_distance, reader2_stress$shortest_distance)

ICC(area_stress)
ICC(shortest_distance_stress)

ba_area_stress <- create_bland_altman_plot(area_stress, "reader1_stress.lumen_area", "reader2_stress.lumen_area", 4.5, -4.5)
ba_shortest_distance_stress <- create_bland_altman_plot(shortest_distance_stress, "reader1_stress.shortest_distance", "reader2_stress.shortest_distance", 1, -1)

ggsave("output/ba_area_stress.png", ba_area_stress, width = 6, height = 4)
ggsave("output/ba_shortest_distance_stress.png", ba_shortest_distance_stress, width = 6, height = 4)