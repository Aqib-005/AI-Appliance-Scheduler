<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateSchedulesTable extends Migration
{
    public function up()
    {
        Schema::create('schedules', function (Blueprint $table) {
            $table->id();
            $table->string('day'); // Day of the week (e.g., Monday)
            $table->foreignId('appliance_id')->constrained()->onDelete('cascade'); // Foreign key to appliances table
            $table->integer('start_hour'); // Start hour (0-23)
            $table->integer('end_hour'); // End hour (0-23)
            $table->timestamps(); // Created at and updated at timestamps
        });
    }

    public function down()
    {
        Schema::dropIfExists('schedules');
    }
}
