<!DOCTYPE html>
<html>

<head>
    <title>Schedule Appliances</title>
</head>

<body>
    <h1>Schedule Appliances</h1>
    <form action="{{ route('schedule.store') }}" method="POST">
        @csrf
        <label for="start_date">Start Date:</label>
        <input type="date" id="start_date" name="start_date" required>
        <button type="submit">Get Predictions</button>
    </form>
</body>

</html>