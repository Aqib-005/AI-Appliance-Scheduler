<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateSchedulesTable extends Migration
{
    public function up()
    {
        if (!Schema::hasTable('schedules')) {
            Schema::create('schedules', function (Blueprint $table) {
                $table->id();
                $table->foreignId('appliance_id')->constrained()->onDelete('cascade');
                $table->string('day');
                $table->integer('start_hour');
                $table->integer('end_hour');
                $table->timestamps();
            });
        }
    }

    public function down()
    {
        Schema::dropIfExists('schedules');
    }
}