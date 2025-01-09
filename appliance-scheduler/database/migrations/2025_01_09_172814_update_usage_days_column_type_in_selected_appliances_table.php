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
            $table->json('usage_days')->change(); // Change column type to JSON
        });
    }

    public function down()
    {
        Schema::table('selected_appliances', function (Blueprint $table) {
            $table->longText('usage_days')->change(); // Revert to longtext if needed
        });
    }
};
