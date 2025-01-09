<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    /**
     * Run the migrations.
     */
    public function up()
    {
        Schema::table('selected_appliances', function (Blueprint $table) {
            $table->time('predicted_start_time')->nullable()->default(null); // Predicted start time
            $table->time('predicted_end_time')->nullable()->default(null);   // Predicted end time
        });
    }

    public function down()
    {
        Schema::table('selected_appliances', function (Blueprint $table) {
            $table->dropColumn('predicted_start_time');
            $table->dropColumn('predicted_end_time');
        });
    }
};
