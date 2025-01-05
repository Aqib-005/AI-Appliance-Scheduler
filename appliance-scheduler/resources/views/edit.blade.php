<!DOCTYPE html>
<html>

<head>
    <title>Edit Appliance</title>
</head>

<body>
    <h1>Edit Appliance</h1>
    <a href="{{ route('appliances.manage') }}">
        <button>Back to Manage Appliances</button>
    </a>

    <form action="{{ route('appliance.update', $appliance->id) }}" method="POST">
        @csrf
        @method('PUT')
        <label for="name">Appliance Name:</label>
        <input type="text" id="name" name="name" value="{{ $appliance->name }}" required>
        <br>
        <label for="power">Power (kW):</label>
        <input type="number" step="0.1" id="power" name="power" value="{{ $appliance->power }}" required>
        <br>
        <label for="preferred_start">Preferred Start Hour (0-23):</label>
        <input type="number" id="preferred_start" name="preferred_start" value="{{ $appliance->preferred_start }}"
            min="0" max="23" required>
        <br>
        <label for="preferred_end">Preferred End Hour (0-23):</label>
        <input type="number" id="preferred_end" name="preferred_end" value="{{ $appliance->preferred_end }}" min="0"
            max="23" required>
        <br>
        <label for="duration">Duration (hours):</label>
        <input type="number" step="0.1" id="duration" name="duration" value="{{ $appliance->duration }}" required>
        <br>
        <button type="submit">Update Appliance</button>
    </form>
</body>

</html>