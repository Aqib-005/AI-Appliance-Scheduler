<!DOCTYPE html>
<html>

<head>
    <meta name="csrf-token" content="{{ csrf_token() }}">
    <title>Manage Appliances</title>
    <style>
        :root {
            --border: 1px solid #ccc;
            --bg-header: #f5f5f5;
            --bg-cell: #fff;
            --text: #000;

            --btn-padding: 10px 20px;
            --btn-radius: 5px;
            --btn-font: 1em;

            --color-primary: #007bff;
            --color-primary-hover: #0056b3;
            --color-success: #218838;
            --color-success-hover: #1e7e34;
        }

        body {
            margin: 0;
            padding: 20px;
            background: var(--bg-cell);
            color: var(--text);
            font-family: Arial, sans-serif;
        }

        /* === Header === */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .header h1 {
            margin: 0;
        }

        /* === Buttons === */
        .btn {
            padding: var(--btn-padding);
            border: none;
            border-radius: var(--btn-radius);
            cursor: pointer;
            font-size: var(--btn-font);
        }

        .btn-primary {
            background-color: var(--color-primary);
            color: #fff;
        }

        .btn-primary:hover {
            background-color: var(--color-primary-hover);
        }

        .btn-success {
            background-color: var(--color-success);
            color: #fff;
        }

        .btn-success:hover {
            background-color: var(--color-success-hover);
        }

        /* === Action row === */
        .button-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        /* === Table === */
        table {
            width: 100%;
            border-collapse: collapse;
        }

        th,
        td {
            border: var(--border);
            padding: 8px;
            text-align: left;
        }

        th {
            background-color: var(--bg-header);
        }

        /* === Popups & overlay === */
        .popup {
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: var(--bg-cell);
            padding: 20px;
            border-radius: var(--btn-radius);
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            z-index: 1000;
            width: 320px;
            max-width: 90%;
        }

        .overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.5);
            z-index: 999;
        }

        .popup.active,
        .overlay.active {
            display: block;
        }

        /* simple form spacing */
        .popup label {
            display: block;
            margin-top: 10px;
        }

        .popup input {
            width: 100%;
            padding: 6px;
            margin-top: 4px;
            box-sizing: border-box;
        }

        .popup .actions {
            margin-top: 20px;
            text-align: right;
        }

        .popup .actions .btn {
            margin-left: 10px;
        }
    </style>
</head>

<body>
    <div class="header">
        <h1>Manage Appliances</h1>
        <a href="{{ route('dashboard') }}">
            <button class="btn btn-success">Back to Dashboard</button>
        </a>
    </div>

    <div class="button-container">
        <h2>Existing Appliances</h2>
        <button class="btn btn-success" onclick="openAddPopup()">Add Appliance</button>
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
                        <button class="btn btn-primary" onclick="openEditPopup({{ $appliance->id }})">Edit</button>
                        <button class="btn btn-primary" onclick="openDeletePopup({{ $appliance->id }})">Delete</button>
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

            <label for="power">Power (kW):</label>
            <input type="number" step="0.1" id="power" name="power" required>

            <label for="preferred_start">Preferred Start Time:</label>
            <input type="time" id="preferred_start" name="preferred_start" required>

            <label for="preferred_end">Preferred End Time:</label>
            <input type="time" id="preferred_end" name="preferred_end" required>

            <label for="duration">Duration (hours):</label>
            <input type="number" step="0.1" id="duration" name="duration" required>

            <div class="actions">
                <button type="submit" class="btn btn-primary">Add</button>
                <button type="button" class="btn" onclick="closeAddPopup()">Cancel</button>
            </div>
        </form>
    </div>

    <!-- Edit Appliance Popup -->
    <div id="editPopup" class="popup">
        <h2>Edit Appliance</h2>
        <form id="editForm" method="POST">
            @csrf
            @method('PUT')

            <label for="edit_name">Appliance Name:</label>
            <input type="text" id="edit_name" name="name" required>

            <label for="edit_power">Power (kW):</label>
            <input type="number" step="0.1" id="edit_power" name="power" required>

            <label for="edit_preferred_start">Preferred Start Time:</label>
            <input type="time" id="edit_preferred_start" name="preferred_start" required>

            <label for="edit_preferred_end">Preferred End Time:</label>
            <input type="time" id="edit_preferred_end" name="preferred_end" required>

            <label for="edit_duration">Duration (hours):</label>
            <input type="number" step="0.1" id="edit_duration" name="duration" required>

            <div class="actions">
                <button type="submit" class="btn btn-primary">Update</button>
                <button type="button" class="btn" onclick="closeEditPopup()">Cancel</button>
            </div>
        </form>
    </div>

    <!-- Delete Confirmation Popup -->
    <div id="deletePopup" class="popup">
        <h2>Delete Appliance</h2>
        <p>Are you sure you want to delete this appliance?</p>
        <form id="deleteForm" method="POST">
            @csrf
            @method('DELETE')
            <div class="actions">
                <button type="submit" class="btn btn-success">Yes</button>
                <button type="button" class="btn" onclick="closeDeletePopup()">No</button>
            </div>
        </form>
    </div>

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
            const appliance = {!! json_encode($appliances->keyBy('id')->toArray()) !!}[applianceId];
            document.getElementById('edit_name').value = appliance.name;
            document.getElementById('edit_power').value = appliance.power;
            document.getElementById('edit_preferred_start').value = appliance.preferred_start?.substring(0, 5);
            document.getElementById('edit_preferred_end').value = appliance.preferred_end?.substring(0, 5);
            document.getElementById('edit_duration').value = appliance.duration;
            document.getElementById('editForm').action = `/appliance/edit/${applianceId}`;

            document.getElementById('editPopup').classList.add('active');
            document.getElementById('overlay').classList.add('active');
        }
        function closeEditPopup() {
            document.getElementById('editPopup').classList.remove('active');
            document.getElementById('overlay').classList.remove('active');
        }

        // Delete Confirmation Popup
        function openDeletePopup(applianceId) {
            document.getElementById('deleteForm').action = `/appliance/remove/${applianceId}`;
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