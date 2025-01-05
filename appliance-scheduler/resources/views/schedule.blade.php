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

    <h2>Manage Appliances</h2>
    <form action="{{ route('appliance.add') }}" method="POST">
        @csrf
        <label for="name">Appliance Name:</label>
        <input type="text" id="name" name="name" required>
        <br>
        <label for="power">Power (kW):</label>
        <input type="number" step="0.1" id="power" name="power" required>
        <br>
        <label for="preferred_start">Preferred Start Hour (0-23):</label>
        <input type="number" id="preferred_start" name="preferred_start" min="0" max="23" required>
        <br>
        <label for="preferred_end">Preferred End Hour (0-23):</label>
        <input type="number" id="preferred_end" name="preferred_end" min="0" max="23" required>
        <br>
        <label for="duration">Duration (hours):</label>
        <input type="number" step="0.1" id="duration" name="duration" required>
        <br>
        <label>Usage Days:</label>
        @foreach (['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] as $day)
            <input type="checkbox" name="usage_days[]" value="{{ $day }}"> {{ $day }}
        @endforeach
        <br>
        <button type="submit">Add Appliance</button>
    </form>

    <h3>Existing Appliances</h3>
    <ul>
        @foreach ($appliances as $appliance)
            <li>
                {{ $appliance->name }} ({{ $appliance->power }} kW)
                <form action="{{ route('appliance.remove', $appliance->id) }}" method="POST" style="display:inline;">
                    @csrf
                    @method('DELETE')
                    <button type="submit">Remove</button>
                </form>
            </li>
        @endforeach
    </ul>
</body>

</html>