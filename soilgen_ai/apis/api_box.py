import os
from typing import Optional

from box_sdk_gen import BoxClient, BoxDeveloperTokenAuth
from box_sdk_gen.managers.uploads import (
    UploadFileAttributes,
    UploadFileAttributesParentField,
    UploadFileVersionAttributes,
)
from box_sdk_gen.networking.auth import Authentication
from dotenv import load_dotenv

from soilgen_ai.apis.base import APIClients
from soilgen_ai.logging_config import setup_logging

logger = setup_logging()


class BoxAPI(APIClients):
    """A client for interacting with the Box API, including file upload and download functionalities."""

    def __init__(self):
        load_dotenv()
        self.token: Optional[str] = os.getenv("AUTH_BOXSDK_TOKEN")
        assert self.token is not None, "BoxSDK token not found in .env file."
        self.auth: Authentication = BoxDeveloperTokenAuth(token=self.token)
        self.client = BoxClient(auth=self.auth)

    def list_files_in_folder(self, folder_id: str = "0"):
        """List all files in a specified Box folder."""
        items = []
        folder_items = self.client.folders.get_folder_items(folder_id)
        if folder_items.entries:
            for item in folder_items.entries:
                logger.info(f"Item: {item.name} (ID: {item.id})")
                items.append(item)
        return items

    def download_file(self, file_id: str, download_path: str):
        """
        Download a file from Box by streaming it directly to a local path.
        This is memory-efficient and suitable for large files.
        """
        with open(download_path, "wb") as output_file:
            # Pass the file handle to the SDK to stream the download
            self.client.downloads.download_file_to_output_stream(file_id, output_file)
        logger.info(f"File downloaded to {download_path}")

    @staticmethod
    def compute_file_size(file_path: str) -> int:
        """Compute the size of a file in bytes."""
        return os.path.getsize(file_path)

    def check_file_exists(
        self, file_name: str, parent_folder_id: str = "0"
    ) -> Optional[str]:
        """
        Checks if a file with the given name exists in a Box folder using the search API.

        Returns:
            The file ID if the file exists, otherwise None.
        """
        # Using search is more efficient than listing all folder items,
        # especially for folders with many files.
        query = f'"{file_name}"'
        search_results = self.client.search.search_for_content(
            query=query, ancestor_folder_ids=[parent_folder_id]
        )
        if search_results.entries:
            # The search can return partial matches, so we must verify the exact name.
            for item in search_results.entries:
                if item.name == file_name:
                    logger.info(f"File '{file_name}' found with ID: {item.id}")
                    return item.id

        logger.info(f"File '{file_name}' not found in folder ID: {parent_folder_id}")
        return None

    def upload_file(self, file_path: str, parent_folder_id: str = "0") -> None:
        """
        Uploads a file to Box, handling existing files by creating a new version.
        It automatically chooses between simple and chunked uploads based on file size.
        """
        file_size = self.compute_file_size(file_path)
        file_name = os.path.basename(file_path)

        existing_file_id = self.check_file_exists(file_name, parent_folder_id)

        # Box recommends chunked uploads for files > 50 MB, but a lower threshold can be robust.
        CHUNKED_UPLOAD_THRESHOLD = 20 * 1024 * 1024  # 20 MB

        with open(file_path, "rb") as file_stream:
            if existing_file_id:
                logger.info(
                    f"File exists. Uploading a new version for file ID: {existing_file_id}..."
                )
                if file_size < CHUNKED_UPLOAD_THRESHOLD:
                    # Use simple upload for new versions of small files
                    attrs = UploadFileVersionAttributes(name=file_name)
                    uploaded_file = self.client.uploads.upload_file_version(
                        file_id=existing_file_id,
                        attributes=attrs,
                        file=file_stream,
                    )
                else:
                    # Use chunked upload for new versions of large files
                    uploaded_file = self.client.chunked_uploads.create_file_upload_session_for_existing_file(
                        file_id=existing_file_id,
                        file_size=file_size,
                    )

            else:
                if file_size < CHUNKED_UPLOAD_THRESHOLD:
                    # Use simple upload for new small files
                    attrs = UploadFileAttributes(
                        name=file_name,
                        parent=UploadFileAttributesParentField(id=parent_folder_id),
                    )
                    uploaded_file = self.client.uploads.upload_file(
                        attributes=attrs, file=file_stream
                    )
                else:
                    # Use chunked upload for new large files
                    uploaded_file = (
                        self.client.chunked_uploads.create_file_upload_session(
                            file_size=file_size,
                            file_name=file_name,
                            folder_id=parent_folder_id,
                        )
                    )

                logger.info(uploaded_file)


if __name__ == "__main__":
    # Example usage:
    box_api = BoxAPI()

    # List files in the root folder
    print("\n--- Listing files in the root folder ---")
    box_api.list_files_in_folder(folder_id="0")

    # Create a dummy file to upload for testing
    dummy_file_path = "test_upload.txt"
    with open(dummy_file_path, "w") as f:
        f.write("This is a test file for the Box API client.")

    # -- DO a check to see if the file exists ---
    print("\n--- Checking if the file exists after first upload ---")
    box_api.check_file_exists(file_name="Bridge_Presentation_CEE573_BFS.pptx")

    # Upload the dummy file for the first time
    print("\n--- Uploading the dummy file for the first time ---")
    box_api.upload_file(file_path=dummy_file_path, parent_folder_id="0")

    # Upload the dummy file again to test versioning
    ## Modify the content to simulate a new version
    with open(dummy_file_path, "a") as f:
        f.write("\nAdding a new line to simulate a new version.")

    print("\n--- Uploading the dummy file again to test versioning ---")
    box_api.upload_file(file_path=dummy_file_path, parent_folder_id="0")
    # Clean up the dummy file
    os.remove(dummy_file_path)
