```

def CalculateSimilarity(UserMatrix):

    SimilarityMatrix ={} #Matrix storing the similarity between every pair of users

    For every pair of users U1, U2 :
        SimilarityMatrix[U1][U2] = Calculate_Cosine_Similarity(U1, U2)

    return SimilarityMatrix


def PredictUserPreferences(TargetUser, RatingMatrix, UserSimilarityMatrix):

    MovieRating = [] # Weighted Predicted Ratings of All Movies Not Watched by User

    For Movies Not Rated By TargetUser:
        MovieRating[MovieNotRatedByTarget] = Sum [(Similarity[Un][TargetUser] * Rating[MovieName][Un]) for Un != TargetUser] / Sum[(SimilarityMatrix[TargetUser][Un]) for Un != TargetUser]

    return MovieRating


def GenerateRecommedations(targetUser, NumRecommendations, RatingMatrix, UserSimilarityMatrix):
    PredictedPreferences = PredictUserPreferences(targetUser, RatingMatrix, UserSimilarityMatix)

    sortedPreferences = sorted(PredictedPreferences, reversed)

    return sortedPreferences[:NumRecommendations]


if __name__ == "__main__":
    RatingMatrix = Rating Matrix of N x M -> N users and M Movies
    SimilarityMatrix = CalculateSimiarity(UserMatrix)
    TargetUser = "UserName"
    NumRecommendations = int("SomeValue")

    recommendations = GenerateRecommendations(TargetUser, NumRecommendations, RatingMatrix, SimilarityMatrix)
    
```
