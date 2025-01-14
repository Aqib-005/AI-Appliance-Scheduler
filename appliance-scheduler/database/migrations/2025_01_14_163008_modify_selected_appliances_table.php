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
            // Add the 'power' column
            $table->decimal('power', 8, 2)->nullable()->after('duration');

            // Remove the 'predicted_start_time' and 'predicted_end_time' columns
            $table->dropColumn(['predicted_start_time', 'predicted_end_time']);
        });
    }

    public function down()
    {
        Schema::table('selected_appliances', function (Blueprint $table) {
            // Reverse the changes (for rollback)
            $table->dropColumn('power');
            $table->time('predicted_start_time')->nullable();
            $table->time('predicted_end_time')->nullable();
        });
    }
};
