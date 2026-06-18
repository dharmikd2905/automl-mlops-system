# Sample Datasets

## Titanic (Classification)

Upload `train.csv` from https://www.kaggle.com/c/titanic/data

Target column: `Survived`

The system will automatically:
- Drop: Name, Ticket, Cabin, PassengerId (no signal)
- Encode: Sex (male/female -> 0/1), Embarked (S/C/Q -> encoded)
- Fill missing: Age -> median, Embarked -> mode

After training, predict survival like this:
```json
{
    "Pclass": 3,
    "Sex": "male",
    "Age": 24.0,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}
```

Expected output: `0` (Did not survive) or `1` (Survived) with confidence %
