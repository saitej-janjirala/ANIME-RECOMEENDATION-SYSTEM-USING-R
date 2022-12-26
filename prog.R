setwd("C://Users//janji//OneDrive//Documents//Anime Recommendation System")

library("psych")  
library(datasets)
library(dplyr)
library("tidyr")
library(ggplot2)
library("recommenderlab")
library("knitr")
library("kableExtra")

#reading anime data
anime <- read.csv("anime.csv")
glimpse(anime)
#reading user ratings
rating <- read.csv("rating.csv")
glimpse(rating)
summary(anime)
summary(rating)
head(anime,10)
head(rating,10)

unique(anime$type)
unique(anime$episodes)

#cleaning anime dataset
print(length(anime$name))

anime <- anime[!is.na(anime$rating),]
anime <- anime[anime$episodes!='Unknown',]
anime[anime$type=='',] <- "Others"
anime$episodes <- as.integer(as.character(anime$episodes))
print(length(anime$name))

#cleaning ratings dataset
print(length(rating$rating))
rating <-rating[rating$rating!=-1,]
print(length(rating$rating))

typestable <-table(anime$type)
typesdf <- as.data.frame(typestable)

typesdf <- typesdf %>% 
  arrange(desc(Var1)) %>%
  mutate(prop = round(Freq / sum(typesdf$Freq) *100,1)) %>%
  mutate(ypos = cumsum(prop)- 0.5*prop )

typesdf

#different types of anime
ggplot(typesdf, aes(x = "", y = prop, fill = Var1)) +
  geom_bar(width = 1, stat = "identity", color = "white") +
  ggtitle("Type of Anime content")+
  coord_polar("y", start = 0)+
  geom_text(aes(y = ypos, label = prop), color = "black")+
  scale_fill_brewer(palette="Set2") +
  theme_void()

#rating vs members size
ggplot(data=anime, aes(x=members, y=rating))+
  geom_point(size=2, colour='red')+
  ggtitle("Distribution of rating with members size")

#ratings vs no.of episodes
ggplot(data=anime, aes(x=episodes, y=rating)) +
  geom_point(size=2, colour='purple')+
  ggtitle("Distribution of rating with episodes size")

top10 <- head(anime[order(-anime$rating),],10)
top10

#Top 10 anime based on ratings
ggplot(top10, aes(x = name, y = rating, fill = name)) + 
  geom_bar(stat = "identity")+
  ggtitle("Top 10 anime based on ratings") +
  geom_text(aes(label=rating), vjust= -0.6, color="black", size=3.5) +
  theme(legend.position="none") + xlab("Anime") + ylab("Rating")

#Top 10 anime based on members size
top10 <- head(anime[order(-anime$members),],10)
top10

ggplot(top10, aes(x = name, y = members, fill = name)) + 
  geom_bar(stat = "identity")+
  ggtitle("Top 10 anime based on members") +
  geom_text(aes(label=members), vjust= -0.6, color="black", size=3.5) +
  theme(legend.position="none") + xlab("Anime") + ylab("members")

#creating real rating matrix
rating <- rating %>% 
  mutate(
    user_id = as.factor(user_id),
    anime_id = as.factor(anime_id),
    rating = as.integer(rating)
  )

test_matrix <- as(rating,"realRatingMatrix")
test_matrix

#filtering the matrix with users who have watched more than 500 animes and animes which has been rated more than 1000 times
test_matrix <- test_matrix[rowCounts(test_matrix) > 500,colCounts(test_matrix)>1000]
test_matrix

#normalizing realRatingMatrix
normalize(test_matrix)

ratings_df <- as.data.frame(table(as.vector(test_matrix@data@x)))

#Distribution of ratings
ggplot(ratings_df, aes(x = Var1, y = Freq, fill = Var1)) + 
  geom_bar(stat = "identity") + 
  ggtitle("Anime ratings distribution") +
  geom_text(aes(label=Freq), vjust= -0.5, color="black", size=4) +
  theme(legend.position="none") + xlab("Rating") + ylab("Fequency")


# evaluation scheme for splitting data randomly 80% for training and 20% for testing and considering 4 items per user for evaluation
#all items with actual user rating of greater or equal 7 are considered positives in the evaluation processing
eval <- evaluationScheme(test_matrix,train =0.8,method = "split",given = 4,goodRating = 7)
eval
  
#Creating a Recommender using UBCF
recommender_ubcf <- Recommender(data = getData(eval,"train"), method = "UBCF")
recommender_ubcf

#predicting with UBCF Recommender
anime_pred_ubcf <- predict(object = recommender_ubcf, newdata= getData(eval,"known"),n = 10)
anime_pred_ubcf

for(i in anime_pred_ubcf@items[1:10]){
  print(i)
}

#Creating a Recommender using SVD
recommender_svd <- Recommender(data = getData(eval,"train"), method = "SVD")
recommender_svd

#predicting with SVD Recommender
anime_pred_svd <- predict(object = recommender_svd, newdata= getData(eval,"known"),n = 10)
anime_pred_svd

for(i in anime_pred_svd@items[1:10]){
  print(i)
}

forUBCF <- TRUE
#function for printing predicted animes
typeof(anime_pred_svd@items)
recommended_animes <- function(i){
  if(forUBCF){
    p <- anime_pred_ubcf@items[[i]]
  }
  else{
    p <- anime_pred_svd@items[[i]]
  }
  p <- data.frame("id" = as.numeric(p))
  p <- inner_join(p, anime, by = c("id" = "anime_id")) %>% select(name,type,episodes,rating,members,genre)
  return(as.data.frame(p))
}


for_users <- c(1,5,7,9)
lapply(for_users, recommended_animes)

forUBCF <- FALSE
lapply(for_users, recommended_animes)

#calculating accuracy for UBCF
ubcf_pred_acc <- calcPredictionAccuracy(x = anime_pred_ubcf, data = getData(eval, "unknown"), given = 4, goodRating = 7)
ubcf_pred_acc

#calculating accuracy for SVD
svd_pred_acc <- calcPredictionAccuracy(x = anime_pred_svd, data = getData(eval, "unknown"), given = 4, goodRating = 7)
svd_pred_acc

#styling accuracies
kable(rbind(ubcf_pred_acc, svd_pred_acc)) %>% kable_styling(c("striped", "hovered", "bordered"), font_size = 12, full_width = 80) %>% add_header_above(c("Algorithms", "Accracy"=9))

#evaluating models for plotting graphs
models_to_evaluate <- list(
  UBCF = list(name = "UBCF", param = NULL),
  SVD = list(name = "SVD", param = NULL)
)

results <- evaluate(eval, method = models_to_evaluate,n=1:5)

#plotting for True positive ratio and False positive ratio
#The x-axis showing 1 – specificity (= false positive fraction = FP/(FP+TN)) 
#The y-axis showing sensitivity (= true positive fraction = TP/(TP+FN))
plot(results, annotate = T, legend = "topleft") 
title("ROC Curve")

#plotting precision vs recall
#Precision = True Positives / (True Positives + False Positives)
#Recall = True Positives / (True Positives + False Negatives)
plot(results, "prec/rec", annotate = T, legend = "bottomright")
title("Pecision-Recall")



