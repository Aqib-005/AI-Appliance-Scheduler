<!DOCTYPE html>
<html>

<head>
    <title>Manage Appliances</title>
    <style>
        /* Table Styles */
        table {
            width: 100%;
            border-collapse: collapse;
        }

        th,
        td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }

        th {
            background-color: #f2f2f2;
        }

        /* Popup Styles */
        .popup {
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            z-index: 1000;
        }

        .overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 999;
        }

        .popup.active,
        .overlay.active {
            display: block;
        }

        /* Button Styles */
        .button-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        button {
            padding: 10px 20px;
            cursor: pointer;
        }
    </style>
</head>

<body>
    <h1>Manage Appliances</h1>
    <a href="{{ route('dashboard') }}">
        <button>Back to Dashboard</button>
    </a>

    <!-- Button and Table -->
    <div class="button-container">
        <h2>Existing Appliances</h2>
        <button onclick="openAddPopup()">Add Appliance</button>
    </div>

    <table>
        <thead>
            <tr>
                <th>Name</th>
                <th>Power (kW)</th>
                <th>Preferred Start</th>
                <th>Preferred End</th>
                <th>Duration (hours)</th>
                <th>Actions</th>
            </tr>
        </thead>
        <tbody>
            @foreach ($appliances as $appliance)
                <tr>
                    <td>{{ $appliance->name }}</td>
                    <td>{{ $appliance->power }}</td>
                    <td>{{ date('H:i', strtotime($appliance->preferred_start)) }}</td>
                    <td>{{ date('H:i', strtotime($appliance->preferred_end)) }}</td>
                    <td>{{ $appliance->duration }}</td>
                    <td>
                        <button onclick="openEditPopup({{ $appliance->id }})">Edit</button>
                        <button onclick="openDeletePopup({{ $appliance->id }})">Delete</button>
                    </td>
                </tr>
            @endforeach
        </tbody>
    </table>

    <!-- Add Appliance Popup -->
    <div id="addPopup" class="popup">
        <h2>Add Appliance</h2>
        <form action="{{ route('appliance.add') }}" method="POST">
            @csrf
            <label for="name">Appliance Name:</label>
            <input type="text" id="name" name="name" required>
            <br>
            <label for="power">Power (kW):</label>
            <input type="number" step="0.1" id="power" name="power" required>
            <br>
            <label for="preferred_start">Preferred Start Time:</label>
            <input type="time" id="preferred_start" name="preferred_start" required>
            <br>
            <label for="preferred_end">Preferred End Time:</label>
            <input type="time" id="preferred_end" name="preferred_end" required>
            <br>
            <label for="duration">Duration (hours):</label>
            <input type="number" step="0.1" id="duration" name="duration" required>
            <br>
            <button type="submit">Add</button>
            <button type="button" onclick="closeAddPopup()">Cancel</button>
        </form>
    </div>

    <!-- Edit Appliance Popup -->
    <div id="editPopup" class="popup">
        <h2>Edit Appliance</h2>
        <form id="editForm" method="POST">
            @csrf
            @method('PUT') <!-- Add this line to spoof the PUT method -->
            <label for="edit_name">Appliance Name:</label>
            <input type="text" id="edit_name" name="name" required>
            <br>
            <label for="edit_power">Power (kW):</label>
            <input type="number" step="0.1" id="edit_power" name="power" required>
            <br>
            <label for="edit_preferred_start">Preferred Start Time:</label>
            <input type="time" id="edit_preferred_start" name="preferred_start" required>
            <br>
            <label for="edit_preferred_end">Preferred End Time:</label>
            <input type="time" id="edit_preferred_end" name="preferred_end" required>
            <br>
            <label for="edit_duration">Duration (hours):</label>
            <input type="number" step="0.1" id="edit_duration" name="duration" required>
            <br>
            <button type="submit">Update</button>
            <button type="button" onclick="closeEditPopup()">Cancel</button>
        </form>
    </div>

    <!-- Delete Confirmation Popup -->
    <div id="deletePopup" class="popup">
        <h2>Delete Appliance</h2>
        <p>Are you sure you want to delete this appliance?</p>
        <form id="deleteForm" method="POST">
            @csrf
            @method('DELETE')
            <button type="submit">Yes</button>
            <button type="button" onclick="closeDeletePopup()">No</button>
        </form>
    </div>

    <!-- Overlay -->
    <div id="overlay" class="overlay"></div>

    <script>
        // Add Appliance Popup
        function openAddPopup() {
            document.getElementById('addPopup').classList.add('active');
            document.getElementById('overlay').classList.add('active');
        }

        function closeAddPopup() {
            document.getElementById('addPopup').classList.remove('active');
            document.getElementById('overlay').classList.remove('active');
        }

        // Edit Appliance Popup
        function openEditPopup(applianceId) {
            // Fetch appliance data (you can use AJAX or preload data)
            const appliance = {!! json_encode($appliances->keyBy('id')->toArray()) !!}[applianceId];

            // Populate the form
            document.getElementById('edit_name').value = appliance.name;
            document.getElementById('edit_power').value = appliance.power;
            document.getElementById('edit_preferred_start').value = appliance.preferred_start;
            document.getElementById('edit_preferred_end').value = appliance.preferred_end;
            document.getElementById('edit_duration').value = appliance.duration;

            // Set the form action
            document.getElementById('editForm').action = `/appliance/edit/${applianceId}`;

            // Show the popup
            document.getElementById('editPopup').classList.add('active');
            document.getElementById('overlay').classList.add('active');
        }

        function closeEditPopup() {
            document.getElementById('editPopup').classList.remove('active');
            document.getElementById('overlay').classList.remove('active');
        }

        // Delete Confirmation Popup
        function openDeletePopup(applianceId) {
            // Set the form action
            document.getElementById('deleteForm').action = `/appliance/remove/${applianceId}`;

            // Show the popup
            document.getElementById('deletePopup').classList.add('active');
            document.getElementById('overlay').classList.add('active');
        }

        function closeDeletePopup() {
            document.getElementById('deletePopup').classList.remove('active');
            document.getElementById('overlay').classList.remove('active');
        }
    </script>
</body>

</html>